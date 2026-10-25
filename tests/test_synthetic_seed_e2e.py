"""Opt-in seeded workloads against a disposable standalone MemMachine server.

Set LTM100_MEMMACHINE_E2E_URL to run. Stored contents are checked directly;
mock embeddings suffice because these tests do not evaluate search quality.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import uuid

import pytest
import yaml

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
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


def _config(path, *, memories=3, corpus_seed=None):
    options = {
        "base_url": os.environ["LTM100_MEMMACHINE_E2E_URL"],
        "org_prefix": f"seed_{uuid.uuid4().hex}",
        "timeout": 30,
    }
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "synthetic",
                    "memories_per_user": memories,
                    "content_chars": 80,
                    "categories": 3,
                    **({"corpus_seed": corpus_seed} if corpus_seed is not None else {}),
                },
                "backend": {"name": "memmachine", **options},
            }
        )
    )
    return options


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("scenario", ["add-load", "search-load"])
async def test_live_seeded_corpus_matches_single_and_multiprocess_runs(
    tmp_path, seed, scenario
):
    dataset = SyntheticAdapter(memories_per_user=3, content_chars=80, categories=3)
    expected = {
        user: {item.content for item in dataset.memory_stream(user)}
        for user in dataset.users(2, seed=seed)
    }
    snapshots = []
    for procs in (1, 2):
        config = tmp_path / f"config_{procs}.yaml"
        options = _config(config)
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
            "12",
            "--top-k",
            "3",
            "--raw",
            "--output",
            str(output),
            "--no-delete-on-exit",
        ]
        if scenario == "search-load":
            args.extend(
                ["--preingest", "--preingest-items-per-user", "3", "--query-limit", "3"]
            )
        async with MemMachineClient(**options) as client:
            try:
                await _cli(*args)
                summary = json.loads((output / "summary.json").read_text())
                assert summary["meta"]["seed"] == seed
                raw = [
                    json.loads(line)
                    for line in (output / "raw.ndjson").read_text().splitlines()
                ]
                assert len(raw) == 12
                assert all(row["status"] == "ok" for row in raw)
                if scenario == "search-load":
                    assert summary["meta"]["preingest_stats"]["input_items"] == 6
                stored = {}
                for user, contents in expected.items():
                    stored[user] = set()
                    for content in contents:
                        hits = await client.search(user, QueryItem(content, top_k=20))
                        returned = {hit.content for hit in hits}
                        assert content in returned
                        assert returned <= contents
                        stored[user].update(returned)
                assert stored == expected
                snapshots.append(stored)
            finally:
                await client.teardown(list(expected), delete=True)
    assert snapshots[0] == snapshots[1]


async def test_live_seeded_memory_growth_sweep(tmp_path):
    config = tmp_path / "config.yaml"
    _config(config, memories=6, corpus_seed=0)
    output = tmp_path / "sweep"
    await _cli(
        "sweep",
        "memory-growth",
        "--config",
        str(config),
        "--memory-counts",
        "3,6",
        "--queries-per-user",
        "3",
        "--users",
        "2",
        "--procs",
        "2",
        "--seed",
        "42",
        "--ops",
        "12",
        "--repetitions",
        "2",
        "--top-k",
        "3",
        "--raw",
        "--output",
        str(output),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert manifest["config"]["seed"] == 42
    settings = manifest["config"]["dataset_config"]
    assert settings["corpus_seed"] == 0
    assert settings["corpus_seed_source"] == "dataset"
    for point in manifest["points"]:
        for repeat in point["repetitions"]:
            assert repeat["status"] == "ok"
            assert repeat["metrics"]["input_items"] == point["memory_count"] * 2
            payload = json.loads(
                (output / repeat["output"] / "summary.json").read_text()
            )
            assert payload["meta"]["dataset_config"] == settings
            raw = [
                json.loads(line)
                for line in (output / repeat["output"] / "raw.ndjson")
                .read_text()
                .splitlines()
            ]
            assert len(raw) == 12
            assert all(row["status"] == "ok" and row["n_items"] > 0 for row in raw)


@pytest.mark.parametrize("scenario", ["add-load", "search-load", "mixed"])
@pytest.mark.parametrize("model", ["closed", "open"])
@pytest.mark.parametrize("procs", [1, 2])
async def test_live_fixed_corpus_workloads_and_reports(
    tmp_path, scenario, model, procs
):
    reference = SyntheticAdapter(
        memories_per_user=4, content_chars=80, categories=3, corpus_seed=-7
    )
    expected = {
        user: {item.content for item in reference.memory_stream(user)}
        for user in reference.users(2, seed=0)
    }
    snapshots = []
    for run_seed in (7, 19):
        config = tmp_path / f"config_{run_seed}.yaml"
        options = _config(config, memories=4, corpus_seed=-7)
        output = tmp_path / f"out_{run_seed}"
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
            str(run_seed),
            *(["--ops", "16"] if model == "closed" else ["--duration", "0.3"]),
            "--model",
            model,
            "--arrival-rate",
            "100",
            "--session-ops",
            "4",
            "--global-concurrency",
            "2",
            "--queue-bound",
            "64",
            "--top-k",
            "4",
            "--think",
            "0",
            "--preingest",
            "--preingest-items-per-user",
            "4",
            "--warmup",
            "0.02",
            "--raw",
            "--output",
            str(output),
            "--no-delete-on-exit",
            "--time-series-interval",
            "0.02",
            "--server-metrics",
            "--server-metrics-interval",
            "0.02",
        ]
        if scenario == "search-load":
            args.extend(["--query-limit", "4"])
        async with MemMachineClient(**options) as client:
            try:
                await _cli(*args)
                payload = json.loads((output / "summary.json").read_text())
                meta, summary = payload["meta"], payload["summary"]
                assert meta["seed"] == run_seed
                assert meta["ops"] == (16 if model == "closed" else 0)
                assert meta["dataset_config"] == reference.generation_settings(
                    seed=run_seed
                )
                assert meta["preingest_stats"]["input_items"] == 8
                assert summary["total"] == summary["successful"] > 0
                if model == "closed":
                    assert summary["total"] == 16
                assert summary["errors"] == summary["rejected"] == 0
                raw = [
                    json.loads(line)
                    for line in (output / "raw.ndjson").read_text().splitlines()
                ]
                assert len(raw) == summary["total"]
                assert all(row["status"] == "ok" for row in raw)
                assert all(
                    row["n_items"] > 0 for row in raw if row["op_type"] == "search"
                )
                assert (output / "summary.csv").is_file()
                with (output / "timeseries.csv").open() as stream:
                    series = list(csv.DictReader(stream))
                assert sum(
                    int(row["completed"]) for row in series if row["op_type"] == "all"
                ) == len(raw)
                assert payload["server_metrics"]["enabled"] is True
                assert payload["server_metrics"]["status"] == "ok"
                for row in payload["server_metrics"]["rows"]:
                    if "http_request" in row["series"] and "/memories" in row["series"]:
                        kind = (
                            "search" if "/memories/search" in row["series"] else "add"
                        )
                        count = sum(r["op_type"] == kind for r in raw)
                        if count:
                            assert row["delta_count"] == count
                        else:
                            assert row["delta_count"] == 0 or (
                                row["delta_count"] is None
                                and row["note"] in {"not executed", "no data"}
                            )
                assert (output / "server_metrics_timeseries.csv").is_file()
                stored = {}
                for user, contents in expected.items():
                    rows = await client.search(
                        user, QueryItem(next(iter(contents)), top_k=100)
                    )
                    stored[user] = {row.content for row in rows}
                    assert stored[user] == contents
                snapshots.append(stored)
            finally:
                await client.teardown(list(expected), delete=True)
    assert snapshots[0] == snapshots[1] == expected
