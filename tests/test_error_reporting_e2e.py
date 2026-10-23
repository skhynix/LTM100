"""CLI fault injection checks reporting alongside unchanged workload contracts."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime

import pytest
import yaml
from aiohttp import web
from test_nebius_e2e import _cli


@pytest.fixture
async def fault_backend():
    operations = []
    lifecycle = []
    projects = set()

    async def handler(request):
        path = request.path
        body = await request.json() if request.can_read_body else {}
        if path.endswith("/health"):
            return web.json_response({"status": "healthy", "version": "fault-test"})
        if path.endswith("/projects"):
            projects.add((body["org_id"], body["project_id"]))
            lifecycle.append("setup")
            return web.json_response({}, status=201)
        if path.endswith("/projects/delete"):
            projects.discard((body["org_id"], body["project_id"]))
            lifecycle.append("delete")
            return web.Response(status=204)
        if request.method == "DELETE":
            lifecycle.append("delete")
            return web.Response(status=204)
        status = 429 if len(operations) % 3 == 0 else 200
        operations.append(
            {"at": time.time(), "path": path, "status": status, "body": body}
        )
        if status == 429:
            return web.json_response({"detail": "private server message"}, status=429)
        if path.endswith("/search"):
            if path.startswith("/api/v2"):
                return web.json_response(
                    {
                        "content": {
                            "episodic_memory": {
                                "long_term_memory": {
                                    "episodes": [{"uid": "id", "content": "hit"}]
                                },
                                "short_term_memory": {"episodes": []},
                            }
                        }
                    }
                )
            return web.json_response({"results": [{"id": "id", "memory": "hit"}]})
        if path == "/v4/memories":
            return web.json_response(
                {
                    "memories": [
                        {"id": f"id-{i}"} for i, _ in enumerate(body["memories"])
                    ]
                }
            )
        count = len(body.get("messages", []))
        return web.json_response(
            {"results": [{"id": f"id-{i}", "uid": f"id-{i}"} for i in range(count)]}
        )

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield (
            f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
            operations,
            lifecycle,
            projects,
        )
    finally:
        await runner.cleanup()


def _config(tmp_path, url, backend="memmachine", scenario="mixed"):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "private user prompt"},
                            {
                                "role": "assistant",
                                "content": "private assistant answer",
                            },
                        ]
                    ]
                }
            ]
        )
    )
    dataset = (
        {"name": "longmemeval", "path": str(source)}
        if scenario == "chat-replay"
        else {"name": "synthetic", "memories_per_user": 4, "content_chars": 200}
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": dataset,
                "backend": {
                    "name": backend,
                    "base_url": url,
                    "timeout": 3,
                    "retries": 1,
                },
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "scenario", ["add-load", "search-load", "mixed", "chat-replay"]
)
@pytest.mark.parametrize("model", ["closed", "open"])
@pytest.mark.parametrize("procs", [1, 2])
async def test_cli_fault_matrix_preserves_budgets_windows_cleanup_and_reports(
    tmp_path, fault_backend, scenario, model, procs
):
    url, operations, lifecycle, projects = fault_backend
    config = _config(tmp_path, url, scenario=scenario)
    output = tmp_path / "out"
    args = [
        "run",
        "--config",
        str(config),
        "--scenario",
        scenario,
        "--users",
        "2",
        "--procs",
        str(procs),
        "--warmup",
        ".03",
        "--raw",
        "--top-k",
        "2",
        "--time-series-interval",
        ".02",
        "--output",
        str(output),
    ]
    if model == "closed":
        args += ["--ops", "12"]
    else:
        args += [
            "--model",
            "open",
            "--duration",
            ".08",
            "--arrival-rate",
            "300",
            "--session-ops",
            "3",
            "--global-concurrency",
            "2",
            "--queue-bound",
            "100",
        ]
    await _cli(*args)
    report = json.loads((output / "summary.json").read_text())
    raw = [
        json.loads(line) for line in (output / "raw.ndjson").read_text().splitlines()
    ]
    summary, meta = report["summary"], report["meta"]
    started = datetime.fromisoformat(meta["measurement_started_at"]).timestamp()
    ended = datetime.fromisoformat(meta["measurement_ended_at"]).timestamp()
    assert (
        projects == set() and lifecycle.count("setup") == lifecycle.count("delete") == 2
    )
    assert raw and len(raw) == summary["total"]
    if model == "closed":
        assert len(raw) == 12
    assert summary["errors"] > 0 and summary["successful"] > 0
    measured = [request for request in operations if started <= request["at"] <= ended]
    assert len(measured) == summary["accepted"] == len(raw)
    # Open arrivals can legitimately emit nothing during a short warm-up.
    # Compare actual measured server traffic, not an assumed warm-up count.
    assert sum(request["status"] == 429 for request in measured) == summary["errors"]
    assert summary["errors_by_kind"] == {"RestError": summary["errors"]}
    assert sum(row["count"] for row in summary["error_breakdown"]) == summary["errors"]
    for row in raw:
        assert started <= row["started_at"] <= row["ended_at"] <= ended
        if row["status"] == "error":
            assert (
                row["error_kind"] == "RestError"
                and row["error_detail"]["http_status"] == 429
            )
            assert row["error_detail"]["method"] == "POST"
        else:
            assert row["status"] == "ok" and "error_detail" not in row
    assert "private" not in (output / "raw.ndjson").read_text()
    assert "private" not in json.dumps(summary)
    series = list(csv.DictReader((output / "timeseries.csv").open()))
    assert (
        sum(int(row["completed"]) for row in series if row["op_type"] == "all")
        == summary["accepted"]
    )
    assert (
        sum(int(row["errors"]) for row in series if row["op_type"] == "all")
        == summary["errors"]
    )
    assert "error_breakdown" not in (output / "summary.csv").read_text().splitlines()[0]
    for kind, value in summary["by_op"].items():
        assert value["errors"] == sum(
            row["status"] == "error" and row["op_type"] == kind for row in raw
        )
        successful = [
            row["latency_ms"]
            for row in raw
            if row["op_type"] == kind and row["status"] == "ok"
        ]
        assert value["latency_ms"]["mean"] == pytest.approx(
            sum(successful) / len(successful) if successful else 0
        )


@pytest.mark.parametrize("backend", ["mem0", "supermemory"])
async def test_other_rest_backends_share_diagnostics_without_raw_flag(
    tmp_path, fault_backend, monkeypatch, backend
):
    url, operations, lifecycle, _ = fault_backend
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "private-api-key")
    config = _config(tmp_path, url, backend=backend)
    output = tmp_path / "out"
    await _cli(
        "run",
        "--config",
        str(config),
        "--scenario",
        "mixed",
        "--users",
        "1",
        "--ops",
        "12",
        "--output",
        str(output),
    )
    report = json.loads((output / "summary.json").read_text())
    summary = report["summary"]
    assert not (output / "raw.ndjson").exists()
    assert len(operations) == 12 and lifecycle == ["delete"]
    assert summary["errors"] == 4 and summary["successful"] == 8
    assert sum(row["count"] for row in summary["error_breakdown"]) == 4
    assert all(row["http_status"] == 429 for row in summary["error_breakdown"])
    assert Counter(row["category"] for row in summary["error_breakdown"]) == {
        "http": len(summary["error_breakdown"])
    }
