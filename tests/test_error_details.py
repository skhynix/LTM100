"""Failure diagnostics preserve exceptions, metrics, and legacy report formats."""

from __future__ import annotations

import asyncio
import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.adapters.transports.rest import RestError, RestTransport
from ltm100.core.config import RunConfig
from ltm100.core.op import Op, OpResult, OpType
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import AddLoad
from ltm100.errors import ErrorDetail, annotate_error, error_detail
from ltm100.metrics.aggregate import aggregate
from ltm100.metrics.report import write_raw_ndjson, write_summary_csv
from ltm100.metrics.timeseries import aggregate_timeseries


@pytest.fixture
async def fault_server():
    calls = []

    async def handler(request):
        calls.append((request.method, request.path, await request.text()))
        mode = request.match_info["mode"]
        if mode == "timeout":
            await asyncio.sleep(0.05)
            return web.json_response({})
        if mode == "disconnect":
            request.transport.close()
            return web.Response()
        if mode == "invalid":
            return web.Response(text="private response is not JSON")
        if mode == "ok":
            return web.json_response({"ok": True})
        return web.Response(status=int(mode), text="secret server response")

    app = web.Application()
    app.router.add_route("*", "/{mode}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", calls
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("status", [401, 404, 409, 429, 500, 599])
@pytest.mark.parametrize("text", [False, True])
async def test_http_context_keeps_status_text_and_no_http_retries(
    fault_server, status, text
):
    url, calls = fault_server
    async with RestTransport(url, retries=2) as client:
        with pytest.raises(RestError) as captured:
            if text:
                await client.request_text("GET", f"/{status}?token=private-query")
            else:
                await client.request(
                    "POST",
                    f"/{status}?token=private-query",
                    json={"private": "request body"},
                )
    error = captured.value
    assert str(error).startswith(
        f"{'GET' if text else 'POST'} /{status}?token=private-query -> {status}:"
    )
    assert "secret server response" in str(error)  # legacy exception contract
    detail = error_detail(error).as_dict()
    assert detail["http_status"] == status
    assert detail["category"] == "http"
    assert detail["endpoint"] == f"/{status}"
    assert detail["method"] == ("GET" if text else "POST")
    assert "secret" not in json.dumps(detail) and "private" not in json.dumps(detail)
    assert len(calls) == 1


@pytest.mark.parametrize("text", [False, True])
async def test_timeout_retry_count_and_original_exception(fault_server, text):
    url, calls = fault_server
    async with RestTransport(
        url, timeout=0.01, retries=2, retry_backoff=0.001
    ) as client:
        with pytest.raises(asyncio.TimeoutError) as captured:
            if text:
                await client.request_text("POST", "/timeout")
            else:
                await client.request("POST", "/timeout")
    assert len(calls) == 3
    assert error_detail(captured.value).as_dict() == {
        "category": "timeout",
        "message": "Request timed out",
        "method": "POST",
        "endpoint": "/timeout",
        "timeout_seconds": 0.01,
    }


async def test_connection_and_response_failures(fault_server):
    url, _ = fault_server
    async with RestTransport(url) as client:
        with pytest.raises(aiohttp.ClientConnectionError) as disconnected:
            await client.request("POST", "/disconnect")
        assert error_detail(disconnected.value).category == "connection"
        with pytest.raises(json.JSONDecodeError) as invalid:
            await client.request("POST", "/invalid")
        detail = error_detail(invalid.value).as_dict()
        assert detail["category"] == "response" and detail["http_status"] == 200
        assert detail["endpoint"] == "/invalid"
        assert "private response" not in json.dumps(detail)
        assert await client.request("POST", "/ok") == {"ok": True}


async def test_retry_preserves_exception_identity(monkeypatch):
    error = aiohttp.ClientConnectionError("credentials in an upstream message")
    calls = []
    client = RestTransport("http://unused", retries=1, retry_backoff=0)

    async def fail(*args, **kwargs):
        calls.append(1)
        raise error

    async with client:
        monkeypatch.setattr(client, "_once", fail)
        with pytest.raises(aiohttp.ClientConnectionError) as captured:
            await client.request("POST", "/memories")
    assert captured.value is error and len(calls) == 2
    assert "credentials" not in json.dumps(error_detail(error).as_dict())


@pytest.mark.parametrize(
    "error",
    [
        ValueError("private prompt"),
        RestError("legacy"),
        TimeoutError(),
        ConnectionError(),
        RuntimeError("private endpoint"),
    ],
)
def test_fallback_is_bounded_and_serializable(error):
    annotate_error(
        error, endpoint="https://user:password@host/search?key=secret#fragment"
    )
    detail = error_detail(error)
    assert detail.endpoint == "/search"
    assert pickle.loads(pickle.dumps(detail)) == detail
    encoded = json.dumps(detail.as_dict())
    assert not any(
        secret in encoded for secret in ("password", "secret", "private", "user:")
    )


def test_diagnostics_cannot_replace_a_hostile_exception():
    class HostileError(Exception):
        @property
        def _ltm100_error_detail(self):
            raise RuntimeError("broken diagnostic property")

    error = HostileError("original failure")
    annotate_error(error, endpoint="/search")
    assert error_detail(error).category == "other"
    assert str(error) == "original failure"
    error.__cause__ = error
    assert error_detail(error).category == "other"


@pytest.mark.parametrize("mode", ["timeout", "tool", "unknown", "cancel"])
async def test_mcp_context_and_cancellation(mode):
    pytest.importorskip("fastmcp")
    from ltm100.adapters.transports.mcp import McpError, McpTransport

    original = {
        "timeout": TimeoutError(),
        "unknown": ValueError("private tool args"),
        "cancel": asyncio.CancelledError(),
    }.get(mode)

    class Client:
        async def call_tool(self, name, arguments):
            if original is not None:
                raise original
            return SimpleNamespace(
                data=SimpleNamespace(status=503, message="secret body", content=None)
            )

    client = McpTransport("https://user:pass@host/mcp?key=secret", timeout=3)
    client._client = Client()
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await client.call_tool("search_memory", {"private": "data"})
        return
    with pytest.raises(McpError) as captured:
        await client.call_tool("search_memory", {"private": "data"})
    detail = error_detail(captured.value).as_dict()
    assert detail["endpoint"] == "/mcp" and detail["tool_name"] == "search_memory"
    assert "http_status" not in detail
    assert not any(
        value in json.dumps(detail) for value in ("private", "secret", "user:pass")
    )
    if mode == "timeout":
        assert captured.value.__cause__ is original
        assert detail["category"] == "timeout" and detail["timeout_seconds"] == 3
    elif mode == "tool":
        assert detail["category"] == "mcp" and detail["tool_status"] == 503
    else:
        assert captured.value.__cause__ is original and detail["category"] == "mcp"


async def test_request_end_precedes_diagnostic_work(monkeypatch):
    import ltm100.core.runner as module

    clock = [100.0]

    class Client:
        async def add(self, user, items):
            clock[0] += 0.25
            raise TimeoutError()

    original = module.error_detail

    def slow_diagnostics(error):
        clock[0] += 50
        return original(error)

    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module, "error_detail", slow_diagnostics)
    runner = LoadRunner(
        client=Client(),
        dataset=SyntheticAdapter(),
        scenario=AddLoad(),
        config=RunConfig(users=1, ops=1),
    )
    await runner._execute("u0", Op(OpType.ADD))
    record = runner.recorder.raw()[0]
    assert record.ended_at - record.started_at == 0.25
    assert (
        record.error_kind == "TimeoutError"
        and record.error_detail.category == "timeout"
    )


async def test_runner_cancellation_is_not_an_operation_error():
    class Client:
        async def add(self, user, items):
            raise asyncio.CancelledError()

    runner = LoadRunner(
        client=Client(),
        dataset=SyntheticAdapter(),
        scenario=AddLoad(),
        config=RunConfig(users=1, ops=1),
    )
    with pytest.raises(asyncio.CancelledError):
        await runner._execute("u0", Op(OpType.ADD))
    assert runner.recorder.raw() == []


def _strip_breakdown(summary):
    return {
        key: (
            {k: _strip_breakdown(v) for k, v in value.items()}
            if key in {"by_op", "by_group"}
            else value
        )
        for key, value in summary.items()
        if key != "error_breakdown"
    }


def test_details_preserve_legacy_metrics_csv_raw_and_timeseries(tmp_path):
    detail = ErrorDetail(
        "http",
        "HTTP 429 Too Many Requests",
        method="POST",
        endpoint="/search",
        http_status=429,
    )
    results = [
        OpResult(OpType.SEARCH, "u0", 0, 0.1, "ok", n_items=2, group="standard"),
        OpResult(
            OpType.SEARCH,
            "u1",
            0.2,
            0.3,
            "error",
            error_kind="RestError",
            group="standard",
            error_detail=detail,
        ),
        OpResult(
            OpType.SEARCH,
            "u2",
            0.4,
            0.5,
            "error",
            error_kind="RestError",
            error_detail=detail,
        ),
        OpResult(OpType.ADD, "u0", 0.6, 0.7, "error", error_kind="ValueError"),
        OpResult(OpType.ADD, "u1", 0.8, 0.8, "rejected", error_kind="queue_full"),
    ]
    legacy = [replace(row, error_detail=None) for row in results]
    old, new = aggregate(legacy), aggregate(results)
    assert _strip_breakdown(new) == old
    assert sum(row["count"] for row in new["error_breakdown"]) == new["errors"] == 3
    assert new["by_group"]["standard"]["error_breakdown"][0]["count"] == 1
    assert len(new["by_op"]["search"]["error_breakdown"]) == 1
    assert new["by_op"]["search"]["error_breakdown"][0]["count"] == 2
    for name, value in [("old", old), ("new", new)]:
        write_summary_csv(value, tmp_path / f"{name}.csv")
    assert (tmp_path / "old.csv").read_bytes() == (tmp_path / "new.csv").read_bytes()
    for name, value in [("old", legacy), ("new", results)]:
        write_raw_ndjson(value, tmp_path / f"{name}.ndjson")
    old_raw = [
        json.loads(line) for line in (tmp_path / "old.ndjson").read_text().splitlines()
    ]
    new_raw = [
        json.loads(line) for line in (tmp_path / "new.ndjson").read_text().splitlines()
    ]
    assert [
        {k: v for k, v in row.items() if k != "error_detail"} for row in new_raw
    ] == old_raw
    assert "error_detail" not in new_raw[0] and "error_detail" not in new_raw[-1]
    assert aggregate_timeseries(
        results, interval_seconds=0.2, measurement_started_at=0, measurement_ended_at=1
    ) == aggregate_timeseries(
        legacy, interval_seconds=0.2, measurement_started_at=0, measurement_ended_at=1
    )
    assert pickle.loads(pickle.dumps(results)) == results
