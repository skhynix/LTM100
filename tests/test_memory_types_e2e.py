"""Opt-in real semantic extraction, indexing, search, and tenant cleanup.

The disposable server must enable semantic memory and use a controlled LLM
that extracts LTM100_FACT markers into features (not an empty chat stub).
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import uuid
from itertools import product

import pytest
import yaml
from test_nebius_e2e import _cli

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.transports.rest import RestError
from ltm100.common import MemoryItem, QueryItem

pytestmark = pytest.mark.skipif(
    not os.environ.get("LTM100_SEMANTIC_E2E_URL"),
    reason="requires a disposable semantic-enabled MemMachine and controlled extraction LLM",
)
TYPES = [["episodic"], ["semantic"], ["episodic", "semantic"]]


@pytest.mark.parametrize("add_types,search_types", list(product(TYPES, TYPES)))
async def test_live_selected_types_extraction_and_tenant_isolation(
    add_types, search_types
):
    options = {
        "base_url": os.environ["LTM100_SEMANTIC_E2E_URL"],
        "org_prefix": f"types_{uuid.uuid4().hex}",
        "timeout": 30,
    }
    users = ["u0", "u1"]
    facts = {user: f"fact_{uuid.uuid4().hex}" for user in users}
    async with MemMachineClient(
        **options, add_memory_types=add_types, search_memory_types=search_types
    ) as client:
        try:
            await client.setup(users)
            for user in users:
                ids = await client.add(
                    user,
                    [
                        MemoryItem(
                            f"LTM100_FACT:{facts[user]}", producer=user, role="user"
                        )
                    ],
                )
                assert len(ids) == 1
            if "semantic" in add_types:
                # This synchronization belongs to the test fixture, not the runner.
                async with MemMachineClient(
                    **options, search_memory_types=["semantic"]
                ) as reader:

                    async def wait_for_features():
                        while True:
                            values = []
                            for user in users:
                                rows = await reader.search(
                                    user, QueryItem(facts[user], top_k=100)
                                )
                                found = {
                                    r.metadata["semantic_feature"]["value"]
                                    for r in rows
                                }
                                assert not (
                                    found
                                    & {facts[other] for other in users if other != user}
                                )
                                values.append(facts[user] in found)
                            if all(values):
                                return
                            await asyncio.sleep(0.1)

                    await asyncio.wait_for(wait_for_features(), 20)
            for user in users:
                rows = await client.search(user, QueryItem(facts[user], top_k=100))
                semantic = [
                    r
                    for r in rows
                    if r.metadata.get("ltm100_memory_type") == "semantic"
                ]
                episodic = [r for r in rows if r not in semantic]
                assert bool(semantic) == (
                    "semantic" in add_types and "semantic" in search_types
                )
                assert bool(episodic) == (
                    "episodic" in add_types and "episodic" in search_types
                )
                for row in semantic:
                    assert row.uid and row.content == f"test_fact: {facts[user]}"
                    assert row.score is None
                for row in episodic:
                    assert row.content == f"LTM100_FACT:{facts[user]}"
        finally:
            await client.teardown(users, delete=True)
        for user in users:
            org, project = client._tenant(user)
            with pytest.raises(RestError, match="404"):
                await client._transport.request(
                    "POST",
                    "/api/v2/projects/get",
                    json={"org_id": org, "project_id": project},
                )


@pytest.mark.parametrize("model", ["closed", "open"])
@pytest.mark.parametrize("procs", [1, 2])
@pytest.mark.parametrize(
    "scenario", ["add-load", "search-load", "mixed", "chat-replay"]
)
async def test_live_cli_types_preingest_reports_and_processes(
    tmp_path, procs, scenario, model
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "synthetic",
                    "memories_per_user": 4,
                    "content_chars": 200,
                },
                "backend": {
                    "name": "memmachine",
                    "base_url": os.environ["LTM100_SEMANTIC_E2E_URL"],
                    "org_prefix": f"types_cli_{uuid.uuid4().hex}",
                    "add_memory_types": ["episodic", "semantic"],
                    "search_memory_types": ["episodic", "semantic"],
                },
            }
        )
    )
    if scenario == "chat-replay":
        source = tmp_path / "dialogue.json"
        source.write_text(
            json.dumps(
                [
                    {
                        "haystack_sessions": [
                            [
                                {
                                    "role": "user",
                                    "content": "LTM100_FACT:chat_user_fact",
                                },
                                {
                                    "role": "assistant",
                                    "content": "LTM100_FACT:chat_assistant_fact",
                                },
                            ]
                        ]
                    }
                ]
            )
        )
        raw_config = yaml.safe_load(config.read_text())
        raw_config["dataset"] = {"name": "longmemeval", "path": str(source)}
        config.write_text(yaml.safe_dump(raw_config))
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
        ".2",
        "--server-metrics",
        "--server-metrics-interval",
        ".1",
        "--raw",
        "--time-series-interval",
        ".1",
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
            ".3",
            "--arrival-rate",
            "40",
            "--session-ops",
            "4",
            "--global-concurrency",
            "2",
            "--queue-bound",
            "50",
        ]
    if scenario != "add-load":
        args += ["--preingest", "--preingest-items-per-user", "2"]
    await _cli(*args)
    report = json.loads((output / "summary.json").read_text())
    raw = [
        json.loads(line) for line in (output / "raw.ndjson").read_text().splitlines()
    ]
    assert raw and len(raw) == report["summary"]["total"]
    if model == "closed":
        assert len(raw) == 12
    assert all(r["status"] == "ok" for r in raw)
    assert report["server_metrics"]["status"] == "ok"
    for row in raw:
        assert row["started_at"] <= row["ended_at"]
    for server_row in report["server_metrics"]["rows"]:
        if (
            "http_request" in server_row["series"]
            and "/memories" in server_row["series"]
        ):
            kind = "search" if "/memories/search" in server_row["series"] else "add"
            expected = sum(r["op_type"] == kind for r in raw)
            if expected:
                assert server_row["delta_count"] == expected
            else:
                assert server_row["delta_count"] == 0 or (
                    server_row["delta_count"] is None
                    and server_row["note"] == "not executed"
                )
    series = list(csv.DictReader((output / "timeseries.csv").open()))
    assert sum(
        int(row["completed"]) for row in series if row["op_type"] == "all"
    ) == len(raw)
    assert (output / "server_metrics_timeseries.csv").exists()
    assert report["meta"]["memory_types"] == {
        "requested": {
            "add": ["episodic", "semantic"],
            "search": ["episodic", "semantic"],
        },
        "resolved_request_types": {
            "add": ["episodic", "semantic"],
            "search": ["episodic", "semantic"],
        },
    }
    if scenario != "add-load":
        assert report["meta"]["preingest_stats"]["input_items"] == 4
