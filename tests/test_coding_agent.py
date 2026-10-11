"""Policy schedules and agent profiles, independent of backend latency."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from itertools import islice

import pytest

from ltm100.adapters.datasets.nebius import NebiusAdapter
from ltm100.agent import AgentMessage, AgentTask, AgentToolCall
from ltm100.core.agent_profile import AgentProfile, load_agent_profile
from ltm100.core.coding_agent import CodingAgent
from ltm100.core.op import Op, OpType, Wait


def task(index=0):
    return AgentTask(
        f"trace-{index}",
        f"issue-{index}",
        "example/repo",
        (
            AgentMessage("system", "Never ingest instructions"),
            AgentMessage("user", f"Fix bug {index}"),
            AgentMessage(
                "assistant",
                "Inspect source",
                (AgentToolCall("read", "read_file", {"path": "a.py"}),),
            ),
            AgentMessage(
                "tool", f"Source result {index}", tool_call_id="read", name="read_file"
            ),
            AgentMessage(
                "assistant",
                "",
                (AgentToolCall("done", "finish", {"message": f"Fixed bug {index}"}),),
            ),
        ),
        "submit",
        bool(index % 2),
        f"Fixed bug {index}",
    )


class AgentSource:
    name = "agent-fixture"

    def __init__(self, tasks=None):
        self.tasks = tasks if tasks is not None else [task()]

    def users(self, n_users, *, seed=0):
        return [f"u{i}" for i in range(n_users)]

    def task_stream(self, user):
        return iter(self.tasks)

    def memory_stream(self, user):
        return iter(())


def events(profile=None, source_task=None, **kwargs):
    scenario = CodingAgent(profile, **kwargs)
    return list(scenario._task_plan("u0", source_task or task(), 0, 42))


def requests(plan):
    return [event for event in plan if isinstance(event, Op)]


@pytest.mark.parametrize(
    "integration,scope,expected",
    [
        ("hook", "user-facing", ["search", "add", "add"]),
        ("hook", "all-steps", ["search", "add", "add", "search", "add", "add"]),
        ("mcp", "user-facing", ["search", "add"]),
        ("hybrid", "user-facing", ["search", "add", "search", "add", "add"]),
        (
            "hybrid",
            "all-steps",
            ["search", "add", "search", "add", "search", "add", "add", "add"],
        ),
    ],
)
def test_exact_policy_order_roles_and_provenance(integration, scope, expected):
    ops = requests(
        events(
            AgentProfile(
                integration=integration,
                hook_scope=scope,
                search_probability=1,
                add_probability=1,
            )
        )
    )
    assert [op.type.value for op in ops] == expected
    assert all(op.context["trajectory_id"] == "trace-0" for op in ops)
    assert all(op.context["source_resolved"] is False for op in ops)
    assert all(op.context["source_exit_status"] == "submit" for op in ops)
    assert all(op.context["task_occurrence"] == 0 for op in ops)
    assert all(op.context["origin"] in {"hook", "mcp"} for op in ops)
    assert all(
        item.role in {"user", "assistant", "tool"}
        and item.producer == "u0"
        and not item.metadata
        for op in ops
        for item in op.items
    )
    stored = "\n".join(item.content for op in ops for item in op.items)
    assert "Never ingest instructions" not in stored
    assert "Tool call: finish" not in stored
    if integration != "mcp":
        assert stored.count("Fixed bug 0") == 1


def test_wait_order_and_tool_override():
    profile = AgentProfile(
        hook_scope="all-steps",
        agent_step_time=2,
        agent_step_time_variation=0,
        tool_time=9,
        tool_time_variation=0,
        tool_times={"read_file": 3},
    )
    plan = events(profile)
    assert [event.delay for event in plan if isinstance(event, Wait)] == [2, 3, 2, 0]
    assert isinstance(plan[2], Wait) and plan[2].delay == 2
    assert plan[3].type is OpType.ADD and plan[3].items[0].role == "assistant"
    assert isinstance(plan[4], Wait) and plan[4].delay == 3
    assert plan[5].type is OpType.SEARCH


def test_framework_user_feedback_is_not_a_new_human_request():
    original = task()
    modified = replace(
        original,
        messages=original.messages[:-1]
        + (AgentMessage("user", "Please continue working"),)
        + original.messages[-1:],
    )
    ops = requests(events(source_task=modified))
    assert len(ops) == 3
    assert all(
        "Please continue working" not in item.content for op in ops for item in op.items
    )
    all_steps = requests(events(AgentProfile(hook_scope="all-steps"), modified))
    assert len(all_steps) == 8
    assert any(
        item.content == "Please continue working"
        for op in all_steps
        for item in op.items
    )


def test_multiple_calls_unknown_tool_and_completed_step_bundles():
    original = task()
    message = replace(
        original.messages[2],
        tool_calls=(
            AgentToolCall("read", "read_file", {"path": "a.py"}),
            AgentToolCall("other", "unknown_tool", {"x": 3}),
        ),
    )
    modified = replace(
        original,
        messages=original.messages[:2]
        + (
            message,
            original.messages[3],
            AgentMessage(
                "tool", "Other result", tool_call_id="other", name="unknown_tool"
            ),
        )
        + original.messages[-1:],
    )
    plan = events(
        AgentProfile(
            integration="mcp",
            search_probability=1,
            add_probability=1,
            tool_time=4,
            tool_time_variation=0,
            tool_times={"unknown_tool": 7},
        ),
        modified,
    )
    ops = requests(plan)
    assert [op.type for op in ops] == [OpType.SEARCH, OpType.ADD, OpType.ADD]
    assert "Source result 0" in ops[1].items[0].content
    assert "Other result" not in ops[1].items[0].content
    assert "Other result" in ops[2].items[0].content
    assert [e.delay for e in plan if isinstance(e, Wait)] == [0, 4, 7, 0, 0]


def test_interrupted_unobserved_call_does_not_invent_results_or_answer():
    original = task()
    interrupted = replace(
        original,
        messages=original.messages[:3],
        exit_status="timeout",
        final_response=None,
    )
    ops = requests(events(source_task=interrupted))
    assert [op.type.value for op in ops] == ["search", "add"]
    mcp = requests(
        events(
            AgentProfile(integration="mcp", search_probability=1, add_probability=1),
            interrupted,
        )
    )
    assert len(mcp) == 1 and mcp[0].type is OpType.SEARCH
    assert all(op.context["source_exit_status"] == "timeout" for op in ops)


def test_queries_use_known_inputs_and_forward_common_search_options():
    original = task()
    modified = replace(
        original,
        messages=original.messages[:-1]
        + (AgentMessage("assistant", "Think more"),)
        + original.messages[-1:],
    )
    ops = requests(
        events(
            AgentProfile(integration="mcp", search_probability=1, add_probability=0),
            modified,
            top_k=50,
            expand_context=2,
            filter="x=y",
        )
    )
    assert len(ops) == 2
    assert ops[0].query.query == "Fix bug 0"
    assert ops[1].query.query == "Fix bug 0\n\nTool result: read_file\nSource result 0"
    assert all("Fixed bug" not in op.query.query for op in ops)
    assert all(
        (op.query.top_k, op.query.expand_context, op.query.filter) == (50, 2, "x=y")
        for op in ops
    )


def test_chunks_bounded_and_source_untouched():
    source_task = task()
    before = asdict(source_task)
    ops = requests(
        events(
            AgentProfile(hook_scope="all-steps", chunk_chars=8, query_chars=7),
            source_task,
        )
    )
    assert all(len(item.content) <= 8 for op in ops for item in op.items)
    assert all(len(op.query.query) <= 7 for op in ops if op.query)
    assert asdict(source_task) == before
    assert all(op.context["chunk_index"] >= 0 for op in ops if op.items)


def test_independent_policy_and_timing_randomness_and_repetition():
    profile = AgentProfile(
        integration="hybrid", search_probability=0.5, add_probability=0.5
    )
    scenario = CodingAgent(profile)

    def replay(seed, occurrence=0):
        return list(scenario._task_plan("u0", task(), occurrence, seed))

    assert replay(8) == replay(8)
    decisions = [requests(replay(seed)) for seed in range(20)]
    assert (
        len(
            {
                json.dumps([asdict(op) for op in ops], sort_keys=True)
                for ops in decisions
            }
        )
        > 1
    )
    assert (
        len(
            {
                json.dumps(
                    [asdict(op) for op in requests(replay(8, occurrence))],
                    sort_keys=True,
                )
                for occurrence in range(20)
            }
        )
        > 1
    )
    timed = CodingAgent(replace(profile, agent_step_time=5, tool_time=2))
    assert requests(replay(8)) == requests(timed._task_plan("u0", task(), 0, 8))


def test_closed_task_order_wrap_and_gap():
    source = AgentSource([task(0), task(1)])
    scenario = CodingAgent(AgentProfile(task_gap=3, task_gap_variation=0))
    plan = list(islice(scenario.plan("u0", source, {"seed": 0}), 26))
    first_ops = requests(plan)
    assert [op.context["trajectory_id"] for op in first_ops][::3] == [
        "trace-0",
        "trace-1",
        "trace-0",
        "trace-1",
    ]
    assert [op.context["task_occurrence"] for op in first_ops][::3] == [0, 1, 2, 3]
    assert sum(isinstance(event, Wait) and event.delay == 3 for event in plan) == 3


def test_finite_tasks_open_boundary_and_admission_cursor():
    source = AgentSource([task(0), task(1)])
    scenario = CodingAgent(AgentProfile(task_repeat=False, task_gap=99))
    first = scenario.prepare_session("u0", source)
    second = scenario.prepare_session("u0", source)
    assert not scenario.session_available("u0", source)
    assert first["task_occurrence"] == 0 and second["task_occurrence"] == 1
    # Finish the second session first; assignment was already fixed at admission.
    second_plan = list(scenario.plan("u0", source, {"seed": 0, **second}))
    first_plan = list(scenario.plan("u0", source, {"seed": 0, **first}))
    assert {op.context["trajectory_id"] for op in requests(first_plan)} == {"trace-0"}
    assert {op.context["trajectory_id"] for op in requests(second_plan)} == {"trace-1"}
    assert all(event.delay != 99 for event in first_plan + second_plan)
    scenario.reset_phase()
    assert scenario.prepare_session("u0", source)["task_occurrence"] == 0


def test_finite_closed_plan_exhausts_once():
    plan = list(
        CodingAgent(AgentProfile(task_repeat=False)).plan(
            "u0", AgentSource([task(0), task(1)]), {"seed": 0}
        )
    )
    assert len(requests(plan)) == 6


@pytest.mark.parametrize(
    "field,value",
    [
        ("integration", "bad"),
        ("hook_scope", "bad"),
        ("task_repeat", 1),
        ("search_probability", -1),
        ("search_probability", 2),
        ("add_probability", True),
        ("tool_time", float("nan")),
        ("agent_step_time", float("inf")),
        ("task_gap", -1),
        ("agent_step_time_variation", 1.1),
        ("tool_time_variation", -0.1),
        ("task_gap_variation", True),
        ("chunk_chars", 0),
        ("query_chars", True),
        ("query_chars", 1.5),
        ("tool_times", []),
        ("tool_times", {"finish": 3}),
        ("tool_times", {"x": -1}),
    ],
)
def test_profile_validation(field, value):
    with pytest.raises((ValueError, TypeError)):
        AgentProfile(**{field: value})


@pytest.mark.parametrize(
    "payload", ["[]", "null", "version: 2", "surprise: 1", "version: true"]
)
def test_invalid_profile_file(tmp_path, payload):
    path = tmp_path / "profile.yaml"
    path.write_text(payload)
    with pytest.raises((ValueError, TypeError)):
        load_agent_profile(path)


def test_load_profile_metadata_and_idle_policy_rejection(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text("version: 1\nintegration: hybrid\ntool_times:\n  read_file: 2\n")
    profile = load_agent_profile(path)
    assert profile.integration == "hybrid" and profile.tool_times == {"read_file": 2}
    assert profile.metadata()["version"] == 1
    assert load_agent_profile(None) == AgentProfile()
    with pytest.raises(ValueError, match="non-zero"):
        AgentProfile(integration="mcp", search_probability=0, add_probability=0)


def test_nebius_capability_without_dialogue_projection(tmp_path):
    from test_nebius_adapter import _record

    path = tmp_path / "source.json"
    path.write_text(json.dumps([_record()]))
    dataset = NebiusAdapter(path=str(path))
    user = dataset.users(1)[0]
    scenario = CodingAgent()
    scenario.validate_run(dataset, [user], model="closed")
    ops = requests(islice(scenario.plan(user, dataset, {"seed": 0}), 6))
    assert len(ops) == 3
    assert not hasattr(dataset, "turn_stream")


def test_early_assistant_does_not_read_a_future_user_request():
    original = task()
    modified = replace(
        original,
        messages=(AgentMessage("assistant", "Early activity"),) + original.messages[1:],
    )
    ops = requests(
        events(
            AgentProfile(integration="mcp", search_probability=1, add_probability=0),
            modified,
        )
    )
    assert ops[0].query.query == ""
    assert "Fix bug" not in ops[0].query.query
    assert ops[1].query.query == "Fix bug 0"


def test_empty_nebius_assignment_rejected_before_replay(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    source = NebiusAdapter(path=str(path))
    with pytest.raises(ValueError, match="assigned user"):
        CodingAgent().validate_run(source, source.users(1), model="closed")


@pytest.mark.parametrize("variation", [None, 0, 0.25, 1])
def test_modeled_delay_distribution_and_seed_reproducibility(variation):
    profile = AgentProfile(
        agent_step_time=20,
        agent_step_time_variation=variation,
        tool_time=10,
        tool_time_variation=variation,
    )
    scenario = CodingAgent(profile)
    samples = []
    for occurrence in range(100):
        first = list(scenario._task_plan("u0", task(), occurrence, 42))
        assert first == list(scenario._task_plan("u0", task(), occurrence, 42))
        delays = [event.delay for event in first if isinstance(event, Wait)][:3]
        assert all(delay >= 0 for delay in delays)
        for delay, mean in zip(delays, (20, 10, 20)):
            samples.append(delay / mean)
            if variation is not None:
                assert 1 - variation <= delay / mean <= 1 + variation
    if variation is None:
        assert any(sample > 2 for sample in samples)
    elif variation > 0:
        assert len(set(samples)) > 10
    else:
        assert set(samples) == {1}
