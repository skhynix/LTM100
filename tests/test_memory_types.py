"""Memory selection preserves legacy payloads and rejects mislabeled runs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from itertools import product
from pathlib import Path

import pytest
import yaml
from aiohttp import web

from ltm100 import cli
from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.backends.memory_types import memory_type_settings
from ltm100.common import MemoryItem, QueryItem
from ltm100.config import AdapterConfig, build_backend, load_config

TYPES = [["episodic"], ["semantic"], ["episodic", "semantic"]]
FEATURE = {
    "set_id": "set-1",
    "category": "profile",
    "tag": "preferences",
    "feature_name": "drink",
    "value": "tea",
    "metadata": {"id": "feature-1", "citations": ["episode-1"], "other": {"x": 1}},
}
EPISODE = {
    "uid": "episode-1",
    "content": "I like tea",
    "score": 0.9,
    "metadata": {"x": 2},
}


@asynccontextmanager
async def server(*, enabled=True, status=200, semantic=None):
    calls = []

    async def handle(request):
        payload = await request.json() if request.can_read_body else None
        calls.append((request.method, request.path, payload))
        if request.method == "GET":
            return web.json_response({"enabled": enabled}, status=status)
        if request.path.endswith("/search"):
            return web.json_response(
                {
                    "content": {
                        "episodic_memory": {
                            "long_term_memory": {"episodes": [EPISODE]}
                        },
                        "semantic_memory": [FEATURE] if semantic is None else semantic,
                    }
                }
            )
        if request.path.endswith("/memories"):
            return web.json_response({"results": [{"uid": "input-id"}]})
        return web.json_response({})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        yield url, calls
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("add,search", list(product(TYPES, TYPES)))
async def test_independent_types_payload_parser_and_single_preflight(add, search):
    async with (
        server() as (url, calls),
        MemMachineClient(
            url, add_memory_types=add, search_memory_types=search
        ) as client,
    ):
        await client.setup(["u0"])
        assert await client.add("u0", [MemoryItem("I like tea", role="user")]) == [
            "input-id"
        ]
        rows = await client.search("u0", QueryItem("drink", top_k=1))
        await client.search("u0", QueryItem("drink"))
        assert calls[-3][2]["types"] == add
        assert calls[-2][2]["types"] == search
        assert len(rows) == len(search)  # Both types may each return top_k.
        if "episodic" in search:
            assert rows[0].content == EPISODE["content"]
            assert rows[0].metadata == EPISODE["metadata"]
        if "semantic" in search:
            row = rows[-1]
            assert row.content == "drink: tea" and row.uid == "feature-1"
            assert row.score is None
            assert row.metadata == {
                "ltm100_memory_type": "semantic",
                "semantic_feature": FEATURE,
            }
        probes = [c for c in calls if c[0] == "GET"]
        assert len(probes) == int("semantic" in add + search)


async def test_legacy_payloads_and_parser_unchanged_without_config_probe():
    async with server(enabled=False) as (url, calls), MemMachineClient(url) as client:
        await client.setup(["u0"])
        await client.add("u0", [MemoryItem("hello", role="user", producer="u0")])
        rows = await client.search("u0", QueryItem("q", top_k=3))
        assert calls == [
            (
                "POST",
                "/api/v2/projects",
                {
                    "org_id": "ltm100",
                    "project_id": "user_u0",
                    "description": "ltm100 virtual user u0",
                },
            ),
            (
                "POST",
                "/api/v2/memories",
                {
                    "org_id": "ltm100",
                    "project_id": "user_u0",
                    "types": ["episodic"],
                    "messages": [
                        {"content": "hello", "producer": "u0", "role": "user"}
                    ],
                },
            ),
            (
                "POST",
                "/api/v2/memories/search",
                {
                    "org_id": "ltm100",
                    "project_id": "user_u0",
                    "query": "q",
                    "top_k": 3,
                    "types": ["episodic"],
                },
            ),
        ]
        assert len(rows) == 1 and rows[0].content == EPISODE["content"]
        assert rows[0].metadata == EPISODE["metadata"]


@pytest.mark.parametrize("option", ["add_memory_types", "search_memory_types"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "semantic",
        True,
        1,
        {},
        ["profile"],
        ["Semantic"],
        [1],
        ["episodic", "episodic"],
        ["episodic", []],
    ],
)
def test_invalid_types_rejected_by_constructor_and_yaml(tmp_path, option, value):
    with pytest.raises(ValueError, match=option):
        MemMachineClient(**{option: value})
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "synthetic"},
                "backend": {"name": "memmachine", option: value},
            }
        )
    )
    with pytest.raises(ValueError, match=option):
        load_config(path)


@pytest.mark.parametrize("backend", ["mem0", "supermemory", "memmachine-mcp"])
@pytest.mark.parametrize("option", ["add_memory_types", "search_memory_types"])
def test_unsupported_backend_rejected_before_construction(backend, option):
    with pytest.raises(ValueError, match=f"{backend} does not support"):
        build_backend(AdapterConfig(backend, {option: ["episodic"]}))


@pytest.mark.parametrize("option", ["add_memory_types", "search_memory_types"])
def test_semantic_producer_isolation_rejected(option):
    with pytest.raises(ValueError, match="filter_by_producer"):
        MemMachineClient(filter_by_producer=True, **{option: ["semantic"]})


@pytest.mark.parametrize("expand,filter", [(1, ""), (0, "x=y")])
async def test_semantic_search_options_rejected_without_request(expand, filter):
    async with (
        server() as (url, calls),
        MemMachineClient(url, search_memory_types=["semantic"]) as client,
    ):
        with pytest.raises(ValueError, match="expand_context or filter"):
            await client.search(
                "u", QueryItem("q", expand_context=expand, filter=filter)
            )
        assert calls == []


@pytest.mark.parametrize(
    "enabled,status",
    [(False, 200), ("true", 200), (None, 200), (True, 403), (True, 404), (True, 500)],
)
async def test_semantic_preflight_failure_has_no_mutation_or_cleanup(
    tmp_path, enabled, status
):
    async with server(enabled=enabled, status=status) as (url, calls):
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "dataset": {"name": "synthetic"},
                    "backend": {
                        "name": "memmachine",
                        "base_url": url,
                        "search_memory_types": ["semantic"],
                    },
                }
            )
        )
        args = cli.build_parser().parse_args(
            [
                "run",
                "--config",
                str(path),
                "--scenario",
                "add-load",
                "--users",
                "1",
                "--ops",
                "1",
            ]
        )
        with pytest.raises((ValueError, RuntimeError), match="semantic"):
            await cli._run_shard_async(args)
        assert len(calls) == 1 and calls[0][:2] == (
            "GET",
            "/api/v2/config/memory/semantic",
        )


async def test_cleanup_does_not_require_semantic_enabled():
    async with (
        server(enabled=False) as (url, calls),
        MemMachineClient(url, search_memory_types=["semantic"]) as client,
    ):
        await client.teardown(["u0"], delete=True)
        assert calls == [
            (
                "POST",
                "/api/v2/projects/delete",
                {"org_id": "ltm100", "project_id": "user_u0"},
            )
        ]


@pytest.mark.parametrize(
    "value",
    [{}, "bad", [None], [{"value": "missing name"}], [{**FEATURE, "metadata": None}]],
)
async def test_invalid_semantic_response_is_not_silent_empty(value):
    async with (
        server(semantic=value) as (url, _calls),
        MemMachineClient(url, search_memory_types=["semantic"]) as client,
    ):
        with pytest.raises(TypeError, match="semantic"):
            await client.search("u0", QueryItem("q"))


def test_omitted_options_are_independent_and_report_is_not_server_readiness():
    settings = memory_type_settings("memmachine", {"add_memory_types": ["semantic"]})
    assert settings == {
        "requested": {"add": ["semantic"], "search": None},
        "resolved_request_types": {"add": ["semantic"], "search": ["episodic"]},
    }


@pytest.mark.parametrize(
    "example",
    ["memmachine.yaml", "synthetic.yaml", "nebius.yaml", "shared-project.yaml"],
)
def test_existing_rest_example_defaults_unchanged(example):
    config = load_config(Path(__file__).parents[1] / "examples" / example)
    assert not (
        {"add_memory_types", "search_memory_types"} & config.backend.options.keys()
    )
    client = build_backend(config.backend)
    assert client.add_memory_types == client.search_memory_types == ("episodic",)


@pytest.mark.parametrize("option,value", [("--expand", "1"), ("--filter", "x=y")])
def test_cli_rejects_unsupported_search_before_backend_probe(
    tmp_path, monkeypatch, option, value
):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "synthetic"},
                "backend": {"name": "memmachine", "search_memory_types": ["semantic"]},
            }
        )
    )

    def forbidden(*args):
        raise AssertionError("backend must not be probed")

    monkeypatch.setattr(cli, "_backend_build", forbidden)
    args = cli.build_parser().parse_args(
        [
            "run",
            "--config",
            str(path),
            "--scenario",
            "search-load",
            "--users",
            "1",
            "--ops",
            "1",
            option,
            value,
        ]
    )
    with pytest.raises(ValueError, match="semantic search"):
        cli._run(args)


async def test_semantic_preflight_rechecked_on_new_connection():
    async with server() as (url, calls):
        client = MemMachineClient(url, search_memory_types=["semantic"])
        for _ in range(2):
            async with client:
                await client.search("u0", QueryItem("q"))
        assert len([c for c in calls if c[0] == "GET"]) == 2
