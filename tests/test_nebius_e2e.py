"""Opt-in Nebius corpus workloads against disposable standalone MemMachine.

Use LTM100_MEMMACHINE_E2E_URL to enable. LTM100_NEBIUS_E2E_PATH can point
to real local trajectories; otherwise a small authored source is used.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

import pytest
import yaml

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.nebius import NebiusAdapter
from ltm100.common import QueryItem

pytestmark = pytest.mark.skipif(
    not os.environ.get("LTM100_MEMMACHINE_E2E_URL"),
    reason="requires an explicitly configured disposable MemMachine server",
)


async def _cli(*args):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "ltm100.cli",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 90)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    assert process.returncode == 0, (stdout.decode(), stderr.decode())


def _source(tmp_path):
    configured = os.environ.get("LTM100_NEBIUS_E2E_PATH")
    if configured:
        return configured
    records = []
    for i in range(2):
        records.append(
            {
                "trajectory_id": f"live-trace-{i}",
                "instance_id": f"example/repo-{i}",
                "repo": "example/repo",
                "exit_status": "submit",
                "resolved": i % 2,
                "trajectory": [
                    {"role": "user", "content": f"Fix live bug {i}"},
                    {"role": "assistant", "content": f"Read file {i}"},
                    {"role": "assistant", "content": f"Change file {i}"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "done",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps(
                                        {"message": f"Fixed live bug {i}"}
                                    ),
                                },
                            }
                        ],
                    },
                ],
            }
        )
    source = tmp_path / "source.json"
    source.write_text(json.dumps(records))
    return str(source)


@pytest.mark.parametrize("scenario", ["add-load", "search-load", "mixed"])
@pytest.mark.parametrize("seed", [0, 42])
async def test_live_nebius_corpus_isolation_and_reports(tmp_path, scenario, seed):
    source = _source(tmp_path)
    dataset = NebiusAdapter(path=source, length=2)
    expected = {
        user: list(dataset.memory_stream(user))[:4]
        for user in dataset.users(2, seed=seed)
    }
    snapshots = []
    for procs in (1, 2):
        options = {
            "base_url": os.environ["LTM100_MEMMACHINE_E2E_URL"],
            "org_prefix": f"neb_{uuid.uuid4().hex}",
            "timeout": 30,
        }
        config = tmp_path / f"config_{procs}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "dataset": {"name": "nebius", "path": source, "length": 2},
                    "backend": {"name": "memmachine", **options},
                }
            )
        )
        output = tmp_path / f"out_{procs}"
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
            "--seed",
            str(seed),
            "--ops",
            "16",
            "--top-k",
            "3",
            "--raw",
            "--time-series-interval",
            "0.1",
            "--server-metrics",
            "--server-metrics-interval",
            "0.1",
            "--output",
            str(output),
            "--no-delete-on-exit",
        ]
        if scenario != "add-load":
            args.extend(["--preingest", "--preingest-items-per-user", "4"])
        if scenario == "search-load":
            args.extend(["--query-limit", "4"])
        async with MemMachineClient(**options) as client:
            try:
                await _cli(*args)
                report = json.loads((output / "summary.json").read_text())
                assert report["meta"]["dataset"] == "nebius"
                assert report["meta"]["users"] == 2
                assert report["meta"]["ops"] == 16
                assert report["meta"]["seed"] == seed
                raw = [
                    json.loads(line)
                    for line in (output / "raw.ndjson").read_text().splitlines()
                ]
                assert len(raw) == 16
                assert all(row["status"] == "ok" for row in raw)
                assert (output / "timeseries.csv").is_file()
                assert (output / "server_metrics_timeseries.csv").is_file()
                if scenario != "add-load":
                    assert report["meta"]["preingest_stats"]["input_items"] == 8
                stored = {}
                for user, items in expected.items():
                    if scenario == "add-load":
                        # The count budget is global, not a per-user minimum.
                        # Backend pacing can leave a user before its fourth add.
                        submitted = sum(
                            row["user_id"] == user and row["op_type"] == "add"
                            for row in raw
                        )
                        items = items[:submitted]
                    expected_contents = {item.content for item in items}
                    stored[user] = set()
                    for item in items:
                        hits = await client.search(
                            user, QueryItem(item.content, top_k=20)
                        )
                        returned = {hit.content for hit in hits}
                        assert item.content in returned
                        if scenario == "search-load":
                            assert returned <= expected_contents
                        # Other workloads may ingest beyond the first four items.
                        other_user_contents = {
                            i.content
                            for u, values in expected.items()
                            if u != user
                            for i in values
                        }
                        assert not returned & (other_user_contents - expected_contents)
                        stored[user].update(returned & expected_contents)
                    assert stored[user] == expected_contents
                snapshots.append(stored)
            finally:
                await client.teardown(list(expected), delete=True)
    if scenario != "add-load":
        # Pre-ingest guarantees this prefix in both process topologies.
        assert snapshots[0] == snapshots[1]
