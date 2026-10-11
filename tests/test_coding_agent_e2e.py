"""Opt-in agent policies against a disposable, real MemMachine standalone.

LTM100_MEMMACHINE_E2E_URL enables the tests. LTM100_NEBIUS_E2E_PATH may
select real local trajectories instead of the authored offline source.
"""

from __future__ import annotations

import csv
import json
import os
import uuid

import pytest
import yaml
from test_nebius_e2e import _cli
from test_nebius_workloads import _write_source

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.nebius import NebiusAdapter
from ltm100.common import QueryItem
from ltm100.core.agent_profile import AgentProfile
from ltm100.core.coding_agent import CodingAgent
from ltm100.core.op import Op, OpType

pytestmark = pytest.mark.skipif(
    not os.environ.get("LTM100_MEMMACHINE_E2E_URL"),
    reason="requires an explicitly configured disposable MemMachine server",
)


@pytest.mark.parametrize(
    "integration,scope",
    [
        ("hook", "user-facing"),
        ("hook", "all-steps"),
        ("mcp", "user-facing"),
        ("hybrid", "user-facing"),
        ("hybrid", "all-steps"),
    ],
)
@pytest.mark.parametrize("model", ["closed", "open"])
async def test_live_agent_policies_task_provenance_and_observation(
    tmp_path, integration, scope, model
):
    source = os.environ.get("LTM100_NEBIUS_E2E_PATH")
    if not source:
        path = tmp_path / "source.json"
        _write_source(path, 2)
        source = str(path)
    dataset = NebiusAdapter(path=source, length=2)
    users = dataset.users(2, seed=42)
    assigned = {user: next(dataset.task_stream(user)) for user in users}
    profile = AgentProfile(
        integration=integration,
        hook_scope=scope,
        search_probability=1,
        add_probability=1,
        agent_step_time=0.002,
        agent_step_time_variation=0,
        tool_time=0.002,
        tool_time_variation=0,
        task_gap=0.002,
        task_gap_variation=0,
    )
    profile_path = tmp_path / "agent.yaml"
    profile_path.write_text(yaml.safe_dump(profile.metadata()))
    for procs in (1, 2):
        options = {
            "base_url": os.environ["LTM100_MEMMACHINE_E2E_URL"],
            "org_prefix": f"agent_{uuid.uuid4().hex}",
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
            "coding-agent",
            "--agent-profile",
            str(profile_path),
            "--users",
            "2",
            "--seed",
            "42",
            "--procs",
            str(procs),
            "--top-k",
            "3",
            "--warmup",
            ".15",
            "--preingest",
            "--preingest-items-per-user",
            "2",
            "--raw",
            "--time-series-interval",
            ".1",
            "--server-metrics",
            "--server-metrics-interval",
            ".1",
            "--output",
            str(output),
            "--no-delete-on-exit",
        ]
        if model == "closed":
            args += ["--ops", "24"]
        else:
            args += [
                "--model",
                "open",
                "--duration",
                ".25",
                "--arrival-rate",
                "30",
                "--session-ops",
                "20",
                "--global-concurrency",
                "2",
                "--queue-bound",
                "20",
            ]
        async with MemMachineClient(**options) as client:
            try:
                await _cli(*args)
                report = json.loads((output / "summary.json").read_text())
                rows = [
                    json.loads(line)
                    for line in (output / "raw.ndjson").read_text().splitlines()
                ]
                assert rows and all(row["status"] == "ok" for row in rows)
                if model == "closed":
                    assert len(rows) == 24
                assert report["meta"]["agent_profile"] == profile.metadata()
                assert report["summary"]["total"] == len(rows)
                assert report["meta"]["preingest_stats"]["input_items"] == 4
                assert report["server_metrics"]["status"] == "ok"
                for server_row in report["server_metrics"]["rows"]:
                    name = server_row["series"]
                    if "http_request" in name and "/memories" in name:
                        kind = "search" if "/memories/search" in name else "add"
                        assert server_row["delta_count"] == sum(
                            row["op_type"] == kind for row in rows
                        )
                series = list(csv.DictReader((output / "timeseries.csv").open()))
                assert sum(
                    int(row["completed"]) for row in series if row["op_type"] == "all"
                ) == len(rows)
                server_series = list(
                    csv.DictReader((output / "server_metrics_timeseries.csv").open())
                )
                assert server_series and any(
                    row["status"] == "ok" for row in server_series
                )
                if model == "open":
                    sessions = report["summary"]["sessions"]
                    assert sessions["admitted"] > 0 and sessions["rejected"] == 0
                by_task = {}
                for row in rows:
                    context = row["context"]
                    assert (
                        context["trajectory_id"]
                        == assigned[row["user_id"]].trajectory_id
                    )
                    by_task.setdefault(
                        (row["user_id"], context["task_occurrence"]), []
                    ).append(row)
                expected_adds = {user: set() for user in users}
                for (user, occurrence), recorded in by_task.items():
                    expected = [
                        op
                        for op in CodingAgent(profile, top_k=3)._task_plan(
                            user, assigned[user], occurrence, 42
                        )
                        if isinstance(op, Op)
                    ]
                    assert len(recorded) <= len(expected)
                    for row, op in zip(recorded, expected):
                        assert row["op_type"] == op.type.value
                        assert row["context"] == op.context
                        if op.type is OpType.ADD:
                            expected_adds[user].update(
                                item.content for item in op.items
                            )
                # Functional storage/isolation checks outside the measured window.
                # Avoid claiming retrieval quality from deterministic embeddings.
                for user in users:
                    for text in list(expected_adds[user])[:3]:
                        hits = await client.search(user, QueryItem(text, top_k=100))
                        contents = {hit.content for hit in hits}
                        assert text in contents
                        other = set().union(
                            *(
                                values
                                for identity, values in expected_adds.items()
                                if identity != user
                            )
                        )
                        assert not (contents & (other - expected_adds[user]))
            finally:
                await client.teardown(users, delete=True)


@pytest.mark.parametrize("warmup", [0, 0.3])
async def test_live_late_source_failure_drains_before_backend_cleanup(tmp_path, warmup):
    import asyncio

    from ltm100.core.config import RunConfig
    from ltm100.core.runner import LoadRunner

    path = tmp_path / "source.json"
    _write_source(path, 2)
    dataset = NebiusAdapter(path=str(path), length=2)
    original_stream = dataset.task_stream

    def broken_stream(user):
        yield next(original_stream(user))
        raise ValueError("corrupt later source trajectory")

    dataset.task_stream = broken_stream
    calls = []
    completed = []
    active = []

    class ObservedClient(MemMachineClient):
        async def search(self, user, query):
            calls.append("search")
            active.append(user)
            try:
                result = await super().search(user, query)
                await asyncio.sleep(0.05)
                completed.append("search")
                return result
            finally:
                active.remove(user)

        async def add(self, user, items):
            calls.append("add")
            return await super().add(user, items)

        async def teardown(self, users, *, delete=True):
            assert not active
            return await super().teardown(users, delete=delete)

    async with ObservedClient(
        base_url=os.environ["LTM100_MEMMACHINE_E2E_URL"],
        org_prefix=f"agent_failure_{uuid.uuid4().hex}",
        timeout=30,
    ) as client:
        run = LoadRunner(
            client=client,
            dataset=dataset,
            scenario=CodingAgent(),
            config=RunConfig(
                users=1,
                duration=0.3,
                warmup=warmup,
                model="open",
                arrival_rate=1000,
                session_ops=20,
            ),
        )
        try:
            with pytest.raises(ValueError, match="corrupt later source"):
                await asyncio.wait_for(run.run(), 10)
            assert calls == completed == ["search"]
            assert not active
        finally:
            await client.teardown(run.users, delete=True)
        await asyncio.sleep(0.05)
        assert calls == completed == ["search"]
        assert not active
