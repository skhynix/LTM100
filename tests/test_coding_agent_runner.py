"""Agent policy schedules through the existing load and reporting pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_coding_agent import AgentSource, task

from ltm100.cli import _build_scenario, _run_metadata, build_parser
from ltm100.common import MemoryItem, ResultItem
from ltm100.core.agent_profile import AgentProfile
from ltm100.core.coding_agent import CodingAgent
from ltm100.core.config import RunConfig
from ltm100.core.multiproc import run_shards
from ltm100.core.op import Op, OpType, Wait
from ltm100.core.runner import LoadRunner
from ltm100.metrics.report import write_raw_ndjson
from ltm100.metrics.timeseries import aggregate_timeseries


class Backend:
    name = "agent-capture"

    def __init__(self, delay=0, fail=False):
        self.delay = delay
        self.fail = fail
        self.calls = []
        self.active = 0
        self.peak = 0
        self.setup_calls = 0

    async def setup(self, users):
        self.setup_calls += 1

    async def call(self, user, kind, payload):
        self.calls.append((user, kind, payload, time.monotonic()))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("backend failed")
        finally:
            self.active -= 1

    async def add(self, user, items):
        await self.call(user, "add", [asdict(item) for item in items])
        return ["id"] * len(items)

    async def search(self, user, query):
        await self.call(user, "search", asdict(query))
        return [ResultItem(content="hit")]


def runner(profile=None, *, backend=None, source=None, **config):
    return LoadRunner(
        client=backend or Backend(),
        dataset=source or AgentSource([task(0), task(1)]),
        scenario=CodingAgent(profile),
        config=RunConfig(**config),
    )


@pytest.mark.parametrize(
    "integration,scope",
    [
        ("hook", "all-steps"),
        ("hook", "user-facing"),
        ("mcp", "user-facing"),
        ("hybrid", "all-steps"),
        ("hybrid", "user-facing"),
    ],
)
async def test_exact_count_waits_and_raw_context(tmp_path, integration, scope):
    profile = AgentProfile(
        integration=integration,
        hook_scope=scope,
        search_probability=1,
        add_probability=1,
        agent_step_time=0.001,
        agent_step_time_variation=0,
        tool_time=0.001,
        tool_time_variation=0,
    )
    run = runner(profile, users=3, ops=37, global_concurrency=2)
    results = await run.run()
    assert len(results) == 37 and all(r.status == "ok" for r in results)
    assert run.client.peak <= 2
    assert all(r.context["origin"] in {"hook", "mcp"} for r in results)
    path = tmp_path / "raw.ndjson"
    write_raw_ndjson(results, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 37 and all(row["context"] for row in rows)
    series = aggregate_timeseries(
        results,
        interval_seconds=0.1,
        measurement_started_at=run.measurement_started_at,
        measurement_ended_at=run.measurement_ended_at,
    )
    assert sum(row["completed"] for row in series if row["op_type"] == "all") == 37


async def test_warmup_exclusion_and_phase_cursors():
    run = runner(users=1, ops=9, warmup=0.025, backend=Backend(0.001))
    boundaries = []

    async def start():
        boundaries.append(("start", len(run.client.calls)))

    async def end():
        boundaries.append(("end", len(run.client.calls)))

    run.on_measure_start, run.on_measure_end = start, end
    results = await run.run()
    assert len(results) == 9
    assert boundaries[0][1] > 0 and boundaries[1][1] - boundaries[0][1] == 9
    assert [r.context["task_occurrence"] for r in results] == [0] * 3 + [1] * 3 + [
        2
    ] * 3
    assert all(
        run.measurement_started_at
        <= r.started_at
        <= r.ended_at
        <= run.measurement_ended_at
        for r in results
    )


@pytest.mark.parametrize("model", ["closed", "open"])
async def test_wait_deadline_expires_without_dispatch(model):
    profile = AgentProfile(
        integration="mcp",
        search_probability=0,
        add_probability=1,
        agent_step_time=10,
        agent_step_time_variation=0,
    )
    run = runner(
        profile, users=1, duration=0.035, model=model, arrival_rate=500, session_ops=10
    )
    began = time.monotonic()
    assert await run.run() == []
    assert time.monotonic() - began < 0.5
    assert run.client.calls == []


async def test_idle_zero_delay_sessions_cooperate_with_deadline():
    only_finish = replace(task(), messages=(task().messages[1], task().messages[-1]))
    run = runner(
        AgentProfile(integration="mcp", search_probability=1, add_probability=0),
        source=AgentSource([only_finish]),
        users=1,
        duration=0.03,
    )
    ticks = []

    async def observer():
        for _ in range(5):
            await asyncio.sleep(0.001)
            ticks.append(1)

    results, _ = await asyncio.wait_for(asyncio.gather(run.run(), observer()), 1)
    assert results == [] and len(ticks) == 5


async def test_count_ineligible_mcp_policy_fails_before_setup():
    only_finish = replace(task(), messages=(task().messages[1], task().messages[-1]))
    run = runner(
        AgentProfile(integration="mcp", search_probability=1, add_probability=0),
        source=AgentSource([only_finish]),
        users=1,
        ops=1,
    )
    with pytest.raises(ValueError, match="cannot emit"):
        await run.run()
    assert run.client.setup_calls == 0


@pytest.mark.parametrize("kind", ["empty", "capability", "finite-count"])
async def test_invalid_run_fails_before_setup(kind):
    run = runner(users=1, ops=1)
    if kind == "empty":
        run.dataset = AgentSource([])
    elif kind == "capability":
        run.dataset.task_stream = None
    else:
        run.scenario = CodingAgent(AgentProfile(task_repeat=False))
    with pytest.raises((ValueError, TypeError)):
        await run.run()
    assert run.client.setup_calls == 0


async def test_finite_closed_run_preserves_trailing_task_time():
    profile = AgentProfile(
        task_repeat=False,
        agent_step_time=0.015,
        agent_step_time_variation=0,
        tool_time=0.015,
        tool_time_variation=0,
    )
    run = runner(profile, users=1, duration=1)
    began = time.monotonic()
    results = await run.run()
    assert len(results) == 6
    assert time.monotonic() - began >= 0.085
    assert results[-1].context["task_occurrence"] == 1


async def test_open_task_boundary_admission_order_and_exhaustion():
    profile = AgentProfile(task_repeat=False, task_gap=10)
    run = runner(
        profile,
        users=1,
        duration=0.065,
        model="open",
        arrival_rate=500,
        session_ops=100,
    )
    results = await run.run()
    assert len(results) == 6
    assert run.session_stats.admitted == run.session_stats.offered == 2
    assert run.session_stats.rejected == 0
    by_occurrence = {}
    for result in results:
        by_occurrence.setdefault(result.context["task_occurrence"], set()).add(
            result.context["trajectory_id"]
        )
    assert by_occurrence == {0: {"trace-0"}, 1: {"trace-1"}}


async def test_open_session_ops_truncates_one_task_and_next_arrival_advances():
    run = runner(users=1, duration=0.07, model="open", arrival_rate=150, session_ops=1)
    results = await run.run()
    assert len(results) >= 3
    assert all(result.type is OpType.SEARCH for result in results)
    assert [r.context["task_occurrence"] for r in results] == list(range(len(results)))
    assert [r.context["trajectory_id"] for r in results] == [
        f"trace-{i % 2}" for i in range(len(results))
    ]


async def test_waits_do_not_hold_global_request_slot():
    run = runner(
        AgentProfile(agent_step_time=0.03, agent_step_time_variation=0),
        users=2,
        ops=6,
        global_concurrency=1,
    )
    await run.run()
    assert run.client.peak == 1
    first_add = next(
        call for call in run.client.calls if call[0] == "u0" and call[1] == "add"
    )
    other_search = next(
        call for call in run.client.calls if call[0] == "u1" and call[1] == "search"
    )
    assert other_search[3] < first_add[3] + 0.02


async def test_count_budget_finishes_without_waiting_after_last_request():
    run = runner(users=1, ops=1)

    def plan(*args):
        yield Op(OpType.ADD, items=[MemoryItem("x")])
        yield Wait(100)

    run.scenario.plan = plan
    assert len(await asyncio.wait_for(run.run(), 1)) == 1


@pytest.mark.parametrize("model", ["closed", "open"])
async def test_unexpected_plan_failure_is_not_a_successful_report(model):
    run = runner(users=1, duration=0.03, model=model, arrival_rate=200, session_ops=2)

    def plan(*args):
        yield Wait()
        raise RuntimeError("broken agent plan")

    run.scenario.plan = plan
    with pytest.raises(RuntimeError, match="broken agent plan"):
        await run.run()


async def test_backend_errors_and_rejections_preserve_context():
    run = runner(
        users=1,
        duration=0.04,
        model="open",
        arrival_rate=600,
        session_ops=4,
        global_concurrency=1,
        queue_bound=1,
        backend=Backend(0.015, fail=True),
    )
    results = await run.run()
    assert any(r.status == "error" for r in results)
    assert any(r.status == "rejected" for r in results)
    assert all(r.context["trajectory_id"] and r.context["origin"] for r in results)
    assert run.client.peak == 1
    assert run._admitted == 0 and run._global_sem._value == 1


def _entry(args, proc_index):
    async def execute():
        run = runner(
            AgentProfile(
                integration="hybrid", search_probability=0.5, add_probability=0.5
            ),
            users=4,
            ops=80 // args["procs"],
            procs=args["procs"],
            proc_index=proc_index,
            seed=args["seed"],
        )
        results = await run.run()
        Path(args["out"], f"{proc_index}.json").write_text(json.dumps(run.client.calls))
        return results

    return asyncio.run(execute())


@pytest.mark.parametrize("seed", [0, 42, -7])
def test_spawned_shards_preserve_per_user_policy_prefix(tmp_path, seed):
    records = []
    traces = []
    for procs in (1, 2):
        out = tmp_path / str(procs)
        out.mkdir()
        result = run_shards(
            _entry, {"procs": procs, "seed": seed, "out": str(out)}, procs
        )
        assert len(result) == 80
        by_user = {}
        for row in result:
            by_user.setdefault(row.user_id, []).append((row.type.value, row.context))
        records.append(by_user)
        payloads = {}
        for path in out.glob("*.json"):
            for user, kind, payload, _ in json.loads(path.read_text()):
                payloads.setdefault(user, []).append((kind, payload))
        traces.append(payloads)
    for user in records[0]:
        common = min(len(records[0][user]), len(records[1][user]))
        assert records[0][user][:common] == records[1][user][:common]
        assert traces[0][user][:common] == traces[1][user][:common]


def test_cli_profile_validation_and_metadata(tmp_path):
    profile = tmp_path / "agent.yaml"
    profile.write_text("integration: hybrid\nagent_step_time: 2\n")
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "unused",
            "--scenario",
            "coding-agent",
            "--agent-profile",
            str(profile),
            "--ops",
            "10",
            "--top-k",
            "50",
        ]
    )
    scenario = _build_scenario(args)
    assert scenario.profile.integration == "hybrid" and scenario.top_k == 50
    args.scenario = "add-load"
    with pytest.raises(ValueError, match="only valid"):
        _build_scenario(args)
    args.scenario = "coding-agent"
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    meta = _run_metadata(
        args,
        dataset="nebius",
        backend="memmachine",
        build={},
        started_at=now,
        ended_at=now,
    )
    assert meta["agent_profile"]["agent_step_time"] == 2
    assert meta["top_k"] == 50


def test_existing_raw_schema_omits_empty_agent_context(tmp_path):
    from ltm100.core.op import OpResult

    path = tmp_path / "raw.ndjson"
    write_raw_ndjson([OpResult(OpType.ADD, "u0", 1, 2, "ok")], path)
    assert "context" not in json.loads(path.read_text())


async def test_waiting_user_does_not_block_another_user_request():
    run = runner(users=2, ops=2, global_concurrency=1)

    def plan(user, *_):
        yield Wait(0.05 if user == "u0" else 0.005)
        yield Op(OpType.ADD, items=[MemoryItem(user)])

    run.scenario.plan = plan
    await run.run()
    assert [call[0] for call in run.client.calls] == ["u1", "u0"]
    assert run.client.calls[1][3] - run.client.calls[0][3] >= 0.035


@pytest.mark.parametrize("model", ["closed", "open"])
async def test_agent_queued_requests_expire_and_running_request_drains(model):
    run = runner(
        users=2,
        duration=0.025,
        model=model,
        arrival_rate=200,
        session_ops=20,
        global_concurrency=1,
        queue_bound=20,
        backend=Backend(0.08),
    )
    began = time.monotonic()
    results = await run.run()
    assert len(results) == 1
    assert run.client.calls[0][3] < began + 0.025
    assert time.monotonic() - began >= 0.07
    assert results[0].context and results[0].status == "ok"
    assert run._global_sem._value == 1 and run._admitted == 0


async def test_operation_latency_excludes_modeled_agent_and_tool_time():
    run = runner(
        AgentProfile(
            agent_step_time=0.03,
            agent_step_time_variation=0,
            tool_time=0.03,
            tool_time_variation=0,
        ),
        users=1,
        ops=3,
    )
    began = time.monotonic()
    results = await run.run()
    assert time.monotonic() - began >= 0.08
    assert all(result.ended_at - result.started_at < 0.02 for result in results)


@pytest.mark.parametrize("duration,ops,expected", [(1, 100, 6), (1, 4, 4)])
async def test_finite_closed_run_accepts_combined_duration_and_ops(
    duration, ops, expected
):
    run = runner(AgentProfile(task_repeat=False), users=1, duration=duration, ops=ops)
    assert len(await run.run()) == expected
    assert run.client.active == 0


async def test_combined_duration_and_ops_accepts_request_free_mcp_pool():
    only_finish = replace(task(), messages=(task().messages[1], task().messages[-1]))
    run = runner(
        AgentProfile(integration="mcp", search_probability=1, add_probability=0),
        source=AgentSource([only_finish]),
        users=1,
        duration=0.02,
        ops=100,
    )
    assert await asyncio.wait_for(run.run(), 1) == []
    assert run.client.setup_calls == 1


@pytest.mark.parametrize("hook", ["session_available", "prepare_session"])
@pytest.mark.parametrize("phase", ["measurement", "warmup"])
@pytest.mark.parametrize("pending", ["executing", "delay", "wait", "queue"])
async def test_open_preparation_failure_drains_before_teardown(hook, phase, pending):
    backend = Backend(delay=0.04)
    run = runner(
        users=1,
        duration=0.2,
        warmup=0.2 if phase == "warmup" else 0,
        model="open",
        arrival_rate=1000,
        session_ops=10,
        global_concurrency=1,
        queue_bound=10,
        backend=backend,
    )
    failure = RuntimeError("later source preparation failed")
    admissions = []
    active_lanes = {}
    original_admit = run._admit_open_session

    def admit(user, active):
        active_lanes.update({user: active[user]})
        return original_admit(user, active)

    run._admit_open_session = admit
    run.scenario.max_sessions_per_user = lambda user: 100

    def prepare(user, dataset):
        state = {"test_session": len(admissions)}
        admissions.append(state)
        return state

    def plan(user, dataset, state):
        lane = state["test_session"]
        if pending == "delay":
            yield Op(OpType.ADD, items=[MemoryItem("delayed")], delay=0.03)
        elif pending == "wait":
            yield Wait(0.03)
        else:
            yield Op(OpType.ADD, items=[MemoryItem(f"first-{lane}")])
        yield Op(OpType.ADD, items=[MemoryItem("must not dispatch")])

    def ready():
        if pending == "queue":
            return backend.active == 1 and run._admitted >= 2
        if pending == "executing":
            return backend.active == 1
        return bool(admissions)

    def available(user, dataset):
        if hook == "session_available" and ready():
            raise failure
        return True

    def guarded_prepare(user, dataset):
        if hook == "prepare_session" and ready():
            raise failure
        return prepare(user, dataset)

    run.scenario.session_available = available
    run.scenario.prepare_session = guarded_prepare
    run.scenario.plan = plan
    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(run.run(), 1)
    assert caught.value is failure
    assert backend.active == 0
    assert run._admitted == 0
    assert not any(active_lanes.values())
    assert run._stop.is_set()
    # The caller may now safely delete state and close the transport.
    calls_at_teardown = len(backend.calls)
    assert calls_at_teardown == (1 if pending in {"executing", "queue"} else 0)
    await asyncio.sleep(0.05)
    assert len(backend.calls) == calls_at_teardown and backend.active == 0
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_coro().__qualname__ == "LoadRunner._open_session"
    ]


async def test_open_synchronous_plan_failure_releases_reserved_lane():
    run = runner(users=1, duration=0.02, model="open", arrival_rate=500, session_ops=1)
    active_lanes = {}
    original_admit = run._admit_open_session

    def admit(user, active):
        active_lanes.update({user: active[user]})
        return original_admit(user, active)

    def broken_plan(*args):
        raise RuntimeError("synchronous plan construction failed")

    run._admit_open_session = admit
    run.scenario.max_sessions_per_user = lambda user: 1
    run.scenario.plan = broken_plan
    with pytest.raises(RuntimeError, match="synchronous plan"):
        await run.run()
    assert not any(active_lanes.values())
    assert run.client.calls == []


async def test_open_arrival_cancellation_drains_executing_request():
    started = asyncio.Event()
    completed = []

    class ObservedBackend(Backend):
        async def call(self, *args):
            started.set()
            await super().call(*args)
            completed.append(True)

    backend = ObservedBackend(delay=0.04)
    run = runner(
        users=1,
        duration=1,
        model="open",
        arrival_rate=100,
        session_ops=10,
        backend=backend,
    )
    workload = asyncio.create_task(run.run())
    await asyncio.wait_for(started.wait(), 1)
    workload.cancel()
    with pytest.raises(asyncio.CancelledError):
        await workload
    assert backend.active == 0 and completed == [True]
    assert len(backend.calls) == 1
    await asyncio.sleep(0.02)
    assert len(backend.calls) == 1


async def test_late_source_read_failure_leaves_no_live_open_session():
    class BrokenSource(AgentSource):
        def task_stream(self, user):
            yield task()
            raise ValueError("corrupt later trajectory")

    backend = Backend(delay=0.04)
    run = runner(
        source=BrokenSource(),
        backend=backend,
        users=1,
        duration=0.2,
        model="open",
        arrival_rate=1000,
        session_ops=10,
    )
    with pytest.raises(ValueError, match="corrupt later trajectory"):
        await run.run()
    assert backend.active == 0 and len(backend.calls) == 1
    await asyncio.sleep(0.05)
    assert backend.active == 0 and len(backend.calls) == 1
