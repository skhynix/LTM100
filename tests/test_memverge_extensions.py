"""Tests for the MemVerge extensions: sharding, rungs, env expansion, items."""

from __future__ import annotations

import asyncio
import itertools

import pytest

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.common import QueryItem
from ltm100.core import scenarios
from ltm100.core.config import RunConfig
from ltm100.core.scenarios import AddLoad, Mixed, SearchLoad
from ltm100.core.op import OpResult, OpType
from ltm100.metrics.aggregate import aggregate


class _Shardable:
    """Minimal stand-in exposing shard_users against a given config."""

    def __init__(self, cfg):
        self.config = cfg

    from ltm100.core.runner import LoadRunner as _LR

    shard_users = _LR.shard_users


def test_single_proc_is_identity():
    users = [f"u{i}" for i in range(10)]
    cfg = RunConfig(users=10, duration=1.0, procs=1)
    assert _Shardable(cfg).shard_users(users) == users


def test_shards_partition_users_exactly_once():
    users = [f"u{i}" for i in range(12)]
    procs = 4
    seen: list[str] = []
    for i in range(procs):
        cfg = RunConfig(users=12, duration=1.0, procs=procs, proc_index=i)
        shard = _Shardable(cfg).shard_users(users)
        assert shard, "every shard must get work"
        seen.extend(shard)
    assert sorted(seen) == sorted(users)
    assert len(seen) == len(set(seen))


def test_procs_may_not_exceed_users():
    with pytest.raises(ValueError, match="exceeds users"):
        RunConfig(users=2, duration=1.0, procs=4)


def test_proc_index_must_be_in_range():
    with pytest.raises(ValueError, match="proc_index"):
        RunConfig(users=8, duration=1.0, procs=2, proc_index=2)







def _result(op: OpType, n_items: int, status: str = "ok") -> OpResult:
    return OpResult(
        type=op, user_id="u", started_at=0.0, ended_at=0.1,
        status=status, n_items=n_items,
    )


def test_empty_rate_separates_working_search_from_silent_search():
    all_empty = aggregate([_result(OpType.SEARCH, 0) for _ in range(4)])
    assert all_empty["error_rate"] == 0.0
    assert all_empty["by_op"]["search"]["items"]["empty_rate"] == 1.0

    productive = aggregate([_result(OpType.SEARCH, 5) for _ in range(4)])
    assert productive["by_op"]["search"]["items"]["empty_rate"] == 0.0
    assert productive["by_op"]["search"]["items"]["mean"] == 5.0


def test_errored_ops_are_excluded_from_empty_rate():
    mixed = aggregate([
        _result(OpType.SEARCH, 5),
        _result(OpType.SEARCH, 0, status="error"),
    ])
    # the errored op has no result count to speak of; it must not read as empty
    assert mixed["by_op"]["search"]["items"]["empty_rate"] == 0.0
    assert mixed["by_op"]["search"]["errors"] == 1


# -- the build recorded in the report ---------------------------------------

class _Cfg:
    def __init__(self, name="memmachine", options=None):
        self.name = name
        self.options = options or {}


class _Bundle:
    def __init__(self, backend):
        self.backend = backend


def _patch_backend(monkeypatch, factory):
    import ltm100.cli as cli
    monkeypatch.setattr(cli, "build_backend", lambda cfg: factory())


def test_build_version_is_recorded(monkeypatch):
    from ltm100.cli import _backend_build

    class Healthy:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None
        async def health(self):
            return {"status": "healthy", "service": "memmachine", "version": "0.3.9.post1"}

    _patch_backend(monkeypatch, Healthy)
    assert _backend_build(_Bundle(_Cfg())) == {
        "build": "0.3.9.post1", "service": "memmachine",
    }


def test_an_unstamped_build_is_reported_as_such(monkeypatch):
    from ltm100.cli import _backend_build

    class Unstamped:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None
        async def health(self):
            return {"service": "memmachine", "version": "0.0.0"}

    _patch_backend(monkeypatch, Unstamped)
    # 0.0.0 is a real answer and must survive verbatim: it is the signal that an
    # image was built without SCM_VERSION, not a missing value to paper over.
    assert _backend_build(_Bundle(_Cfg()))["build"] == "0.0.0"


def test_a_failed_probe_does_not_break_the_report(monkeypatch):
    from ltm100.cli import _backend_build

    class Unreachable:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None
        async def health(self):
            raise ConnectionError("refused")

    _patch_backend(monkeypatch, Unreachable)
    assert "unavailable" in _backend_build(_Bundle(_Cfg()))["build"]


def test_a_backend_without_health_is_simply_omitted(monkeypatch):
    from ltm100.cli import _backend_build

    class NoHealth:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None

    _patch_backend(monkeypatch, NoHealth)
    assert _backend_build(_Bundle(_Cfg())) == {}


def test_the_report_meta_carries_the_build(tmp_path, monkeypatch, capsys):
    """The probe is only useful if _run actually puts it in the report."""
    import json as _json
    import yaml as _yaml

    import ltm100.cli as cli

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml.safe_dump({
        "dataset": {"name": "synthetic", "length": 2},
        "backend": {"name": "memmachine", "base_url": "http://localhost:8080"},
    }))

    monkeypatch.setattr(cli, "_backend_build",
                        lambda cfg: {"build": "9.9.9+test", "service": "memmachine"})
    monkeypatch.setattr(cli, "run_shards", lambda entry, args, procs: [])

    args = cli.build_parser().parse_args([
        "run", "--config", str(cfg_path), "--scenario", "chat-replay",
        "--users", "2", "--duration", "1",
    ])
    assert cli._run(args) == 0

    meta = _json.loads(capsys.readouterr().out)["meta"]
    assert meta["build"] == "9.9.9+test"
    assert meta["service"] == "memmachine"


# -- whole-run load is divided across shards --------------------------------

def test_shares_sum_to_the_total():
    from ltm100.cli import _split
    for total, procs in ((100, 4), (10, 4), (7, 3), (0, 4), (1, 4)):
        parts = [_split(total, procs, i) for i in range(procs)]
        assert sum(parts) == total, (total, procs, parts)


def test_single_process_is_unchanged():
    from ltm100.cli import _split
    assert _split(100, 1, 0) == 100


def test_sharding_divides_concurrency_rate_ops_and_queue():
    """Each shard runs its own runner, so an undivided budget applies N times."""
    import ltm100.cli as cli

    args = cli.build_parser().parse_args([
        "run", "--config", "x", "--scenario", "chat-replay",
        "--users", "24", "--duration", "10", "--procs", "4",
        "--global-concurrency", "100",
        "--model", "open", "--arrival-rate", "40", "--session-ops", "12",
        "--queue-bound", "20",
    ])
    shares = []
    for i in range(4):
        args.proc_index = i
        cfg = cli._build_run_config(args)
        shares.append(cfg)
    assert sum(c.global_concurrency for c in shares) == 100
    assert sum(c.queue_bound for c in shares) == 20
    assert abs(sum(c.arrival_rate for c in shares) - 40) < 1e-9
    # per-session, not per-run: must NOT be divided
    assert all(c.session_ops == 12 for c in shares)
    # users are partitioned by shard_users, so the config keeps the full count
    assert all(c.users == 24 for c in shares)


def test_length_zero_yields_no_samples(tmp_path):
    """The streaming path and the json.load fallback must agree."""
    import json
    from ltm100.adapters.datasets.longmemeval import LongMemEvalAdapter

    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"haystack_sessions": [[{"role": "user", "content": "a"}]],
                              "haystack_session_ids": ["s1"]}] * 3))
    assert LongMemEvalAdapter(path=str(p), length=0)._load() == []


def test_sharding_divides_a_count_based_budget():
    """--ops caps the whole run, so each shard gets a share of it."""
    import ltm100.cli as cli

    args = cli.build_parser().parse_args([
        "run", "--config", "x", "--scenario", "chat-replay",
        "--users", "8", "--ops", "1000", "--procs", "3",
    ])
    total = 0
    for i in range(3):
        args.proc_index = i
        total += cli._build_run_config(args).ops
    assert total == 1000


# -- server-side search knobs: expand_context and filter ---------------------

def _captured_search_payload(**query_kwargs):
    """Run MemMachineClient.search against a stub transport, return the payload."""
    import asyncio
    from ltm100.adapters.backends.memmachine import MemMachineClient
    from ltm100.common import QueryItem

    seen = {}

    class _Stub:
        headers: dict = {}
        async def request(self, method, path, json=None, params=None):
            seen["path"] = path
            seen["payload"] = json
            return {"content": {"episodic_memory": {"long_term_memory": {"episodes": []}}}}

    c = MemMachineClient()
    c._transport = _Stub()
    asyncio.run(c.search("u0", QueryItem(query="q", **query_kwargs)))
    return seen["payload"]


def test_defaults_omit_both_knobs():
    """A default run's payload must be unchanged by this feature."""
    p = _captured_search_payload()
    assert "expand_context" not in p
    assert "filter" not in p
    assert set(p) == {"org_id", "project_id", "query", "top_k", "types"}


def test_expand_context_is_sent_when_set():
    p = _captured_search_payload(expand_context=3)
    assert p["expand_context"] == 3


def test_filter_is_sent_verbatim_when_set():
    # the platform's filter language is metadata.<field>=<value>, exact match
    p = _captured_search_payload(filter="metadata.category=cat_3")
    assert p["filter"] == "metadata.category=cat_3"
    assert "expand_context" not in p


def test_zero_expand_is_treated_as_off():
    assert "expand_context" not in _captured_search_payload(expand_context=0)


def test_scenarios_propagate_the_knobs_to_every_query():
    """A knob set on the CLI is useless if the scenario drops it."""
    from ltm100.core.scenarios import get_scenario
    from ltm100.adapters.datasets.synthetic import SyntheticAdapter

    ds = SyntheticAdapter(memories_per_user=6)
    for name in ("search-load", "mixed"):
        s = get_scenario(name, expand_context=2, filter="metadata.category=cat_1")
        # plan() loops forever to sustain load; islice or it eats all memory
        ops = list(itertools.islice(s.plan("u0", ds, {"seed": 0}), 12))
        queries = [o.query for o in ops if getattr(o, "query", None) is not None]
        assert queries, f"{name} produced no queries"
        assert all(q.expand_context == 2 for q in queries), name
        assert all(q.filter == "metadata.category=cat_1" for q in queries), name


def test_negative_expand_is_rejected():
    from ltm100.core.scenarios import get_scenario
    for name in ("search-load", "mixed", "chat-replay"):
        with pytest.raises(ValueError, match="expand_context"):
            get_scenario(name, expand_context=-1)


def test_synthetic_categories_give_a_field_to_filter_on():
    from ltm100.adapters.datasets.synthetic import SyntheticAdapter

    off = list(SyntheticAdapter(memories_per_user=10).memory_stream("u0"))
    assert all(m.metadata == {} for m in off), "must be inert when unset"

    on = list(SyntheticAdapter(memories_per_user=100, categories=10).memory_stream("u0"))
    cats = [m.metadata["category"] for m in on]
    assert len(set(cats)) == 10
    assert cats.count("cat_3") == 10          # one value selects ~1/N


def test_mcp_refuses_the_knobs_it_cannot_honour():
    """Silently ignoring them would label a baseline search as filtered."""
    import asyncio
    from ltm100.adapters.backends.memmachine_mcp import MemMachineMcpClient
    from ltm100.common import QueryItem

    c = MemMachineMcpClient.__new__(MemMachineMcpClient)
    c.org_prefix = "t"
    for q in (QueryItem(query="x", expand_context=2),
              QueryItem(query="x", filter="metadata.category=cat_3")):
        with pytest.raises(ValueError, match="MCP backend supports neither"):
            asyncio.run(c.search("u0", q))


def test_mcp_refuses_the_isolation_scope_options_by_name():
    """A bare TypeError about an unexpected keyword does not say which backend to use."""
    from ltm100.adapters.backends.memmachine_mcp import MemMachineMcpClient

    for opts in ({"project_id": "shared"}, {"filter_by_producer": True}):
        with pytest.raises(ValueError, match="MCP backend supports neither project_id"):
            MemMachineMcpClient("http://core:8081", **opts)


def test_mcp_refuses_metadata_it_would_drop():
    import asyncio
    from ltm100.adapters.backends.memmachine_mcp import MemMachineMcpClient
    from ltm100.common import MemoryItem

    c = MemMachineMcpClient.__new__(MemMachineMcpClient)
    c.org_prefix = "t"
    with pytest.raises(ValueError, match="cannot store item metadata"):
        asyncio.run(c.add("u0", [MemoryItem(content="a", metadata={"category": "cat_1"})]))


class _CountingSynthetic(SyntheticAdapter):
    """Counts how many times a plan materializes the user's stream."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.builds = 0

    def memory_stream(self, user):
        self.builds += 1
        return super().memory_stream(user)


def _plan_ops(scenario, dataset, n):
    return [
        (op.type, tuple(getattr(i, "query", getattr(i, "content", None)) for i in op.items))
        for op in itertools.islice(scenario.plan("syn_user_00000", dataset, {"seed": 0}), n)
    ]


@pytest.mark.parametrize(
    "make_scenario",
    [lambda: SearchLoad(top_k=20), lambda: AddLoad(), lambda: Mixed()],
    ids=["search-load", "add-load", "mixed"],
)
def test_plans_build_the_corpus_once_per_user_not_once_per_session(make_scenario):
    """Under --model open a session opens per arrival, and planning runs on the
    event loop; rebuilding the corpus each time starves the loop and wedges the
    run. Guards the cache in _memories/_query_pool."""
    scenarios._MEMO.clear()
    dataset = _CountingSynthetic(memories_per_user=200)
    scenario = make_scenario()
    for _ in range(5):
        _plan_ops(scenario, dataset, 3)
    assert dataset.builds == 1, f"corpus rebuilt {dataset.builds} times across 5 sessions"


@pytest.mark.parametrize(
    "make_scenario",
    [lambda: SearchLoad(top_k=20), lambda: AddLoad(), lambda: Mixed()],
    ids=["search-load", "add-load", "mixed"],
)
def test_caching_does_not_change_the_plan(make_scenario):
    """The cache must be invisible: a cached run and an uncached one have to
    yield the same ops, or it would silently alter the workload."""
    scenarios._MEMO.clear()
    cached = _plan_ops(make_scenario(), SyntheticAdapter(memories_per_user=200), 50)

    real_memories, real_pool = scenarios._memories, scenarios._query_pool
    try:
        scenarios._memories = lambda d, u: list(d.memory_stream(u))
        scenarios._query_pool = lambda sc, d, u: scenarios._content_queries(
            list(d.memory_stream(u)), sc.top_k, sc.expand_context, sc.filter
        )
        uncached = _plan_ops(make_scenario(), SyntheticAdapter(memories_per_user=200), 50)
    finally:
        scenarios._memories, scenarios._query_pool = real_memories, real_pool

    assert cached == uncached


# -- shared-project tenancy --------------------------------------------------


def _mm(**kw):
    return MemMachineClient("http://core:8081", org_prefix="t", **kw)


def _capture(client):
    """Replace the transport so request() records payloads instead of sending."""
    seen: list[tuple[str, str, dict]] = []

    async def _request(method, path, json=None, params=None):
        seen.append((method, path, json or {}))
        return {}

    client._transport.request = _request
    return seen


def test_per_user_projects_are_the_default():
    c = _mm()
    assert c._tenant("alice") == ("t", "user_alice")
    assert c._tenant("bob") == ("t", "user_bob")


def test_project_id_collapses_every_user_onto_one_project():
    c = _mm(project_id="shared")
    assert c._tenant("alice") == ("t", "shared")
    assert c._tenant("bob") == ("t", "shared")


def _absent():
    """What projects/get raises for a project that does not exist yet."""
    from ltm100.adapters.transports.rest import RestError

    return RestError("POST /api/v2/projects/get -> 404: Project does not exist")


def test_a_shared_project_is_created_once_not_once_per_user():
    c = _mm(project_id="shared")
    calls = _failing(c, _absent())
    asyncio.run(c.setup([f"u{i}" for i in range(50)]))
    assert calls == ["/api/v2/projects/get", "/api/v2/projects"], calls


def test_per_user_projects_are_still_created_per_user():
    c = _mm()
    seen = _capture(c)
    asyncio.run(c.setup([f"u{i}" for i in range(7)]))
    assert len([x for x in seen if x[1] == "/api/v2/projects"]) == 7


def test_a_shared_project_is_deleted_once_not_once_per_user():
    c = _mm(project_id="shared")
    calls = _failing(c, _absent())
    users = [f"u{i}" for i in range(50)]
    asyncio.run(c.setup(users))
    asyncio.run(c.teardown(users, delete=True))
    assert calls.count("/api/v2/projects/delete") == 1


def test_a_shared_project_that_already_existed_is_not_deleted_on_exit():
    """projects/get found it, so it predates this run: its corpus is not ours to
    drop, whatever delete-on-exit says. Asking first matters because re-creating
    an identical project answers 201, which would look like this run made it."""
    c = _mm(project_id="shared")
    calls = _failing(c)
    asyncio.run(c.setup(["u0", "u1"]))
    asyncio.run(c.teardown(["u0", "u1"], delete=True))
    assert calls == ["/api/v2/projects/get"], calls


def test_per_user_projects_are_still_deleted_on_exit_after_a_409():
    """Per-user teardown is unchanged: those projects are named after the run's
    own users, and upstream deletes them however they came to exist."""
    from ltm100.adapters.transports.rest import RestError

    c = _mm()
    calls = _failing(c, RestError("POST /api/v2/projects -> 409: Project already exists"))
    asyncio.run(c.setup(["u0"]))
    asyncio.run(c.teardown(["u0"], delete=True))
    assert calls.count("/api/v2/projects/delete") == 1


def _failing(client, *errors):
    """Replace the transport so each call raises the next error, then succeeds."""
    calls: list[str] = []
    queue = list(errors)

    async def _request(method, path, json=None, params=None):
        calls.append(path)
        if queue:
            raise queue.pop(0)
        return {}

    client._transport.request = _request
    client._create_retry_delay = 0
    return calls


def test_a_racing_500_on_the_shared_create_is_retried():
    """Every shard creates the shared project at once, and the server can answer
    one racing create with a 500."""
    from ltm100.adapters.transports.rest import RestError

    c = _mm(project_id="shared")
    calls = _failing(
        c, _absent(), RestError("POST /api/v2/projects -> 500: server internal error")
    )
    asyncio.run(c.setup(["u0", "u1"]))
    assert calls == ["/api/v2/projects/get", "/api/v2/projects", "/api/v2/projects"]


def test_a_shared_create_that_keeps_failing_still_raises():
    from ltm100.adapters.transports.rest import RestError

    c = _mm(project_id="shared")
    _failing(c, _absent(), *[RestError("POST /api/v2/projects -> 500: boom")] * 2)
    with pytest.raises(RestError):
        asyncio.run(c.setup(["u0"]))


def test_a_per_user_create_500_is_not_retried():
    """Per-user projects are never created concurrently, so a 500 there is real."""
    from ltm100.adapters.transports.rest import RestError

    c = _mm()
    calls = _failing(c, RestError("POST /api/v2/projects -> 500: boom"))
    with pytest.raises(RestError):
        asyncio.run(c.setup(["u0"]))
    assert len(calls) == 1


def test_the_producer_scope_is_applied_per_user():
    c = _mm(project_id="shared", filter_by_producer=True)
    seen = _capture(c)
    asyncio.run(c.search("alice", QueryItem(query="q")))
    asyncio.run(c.search("bob", QueryItem(query="q")))
    assert [x[2]["filter"] for x in seen] == [
        "producer_id = 'alice'",
        "producer_id = 'bob'",
    ]


def test_the_producer_scope_composes_with_a_caller_filter():
    c = _mm(project_id="shared", filter_by_producer=True)
    seen = _capture(c)
    asyncio.run(c.search("alice", QueryItem(query="q", filter="m.category = 'cat_3'")))
    assert seen[0][2]["filter"] == "producer_id = 'alice' AND (m.category = 'cat_3')"


def test_the_producer_scope_survives_an_or_in_the_caller_filter():
    """AND binds tighter than OR server-side, so the caller's filter must be
    parenthesised or anything after the OR returns whoever produced it."""
    c = _mm(project_id="shared", filter_by_producer=True)
    seen = _capture(c)
    asyncio.run(
        c.search("alice", QueryItem(query="q", filter="m.category = 'a' OR m.category = 'b'"))
    )
    assert seen[0][2]["filter"] == (
        "producer_id = 'alice' AND (m.category = 'a' OR m.category = 'b')"
    )


def test_a_caller_filter_is_untouched_without_the_scope():
    c = _mm()
    seen = _capture(c)
    asyncio.run(c.search("alice", QueryItem(query="q", filter="m.category = 'cat_3'")))
    assert seen[0][2]["filter"] == "m.category = 'cat_3'"


def test_tenancy_options_are_inert_at_their_defaults():
    """A default run's payload must be byte-identical to one built without them."""
    c = _mm()
    seen = _capture(c)
    asyncio.run(c.search("alice", QueryItem(query="q")))
    assert seen[0][2] == {
        "org_id": "t",
        "project_id": "user_alice",
        "query": "q",
        "top_k": 20,
        "types": ["episodic"],
    }


class _Reached(Exception):
    """Raised by a stub to show the run got that far."""


def _stub_launch(monkeypatch):
    """Stop _run at the shard launch, with nothing sent over the network."""
    from ltm100 import cli

    launched: list[int] = []

    def _shards(entry, args, procs):
        launched.append(procs)
        raise _Reached

    monkeypatch.setattr(cli, "_backend_build", lambda cfg: {})
    monkeypatch.setattr(cli, "run_shards", _shards)
    return launched


def _run_args(tmp_path, backend_extra, *flags):
    from ltm100.cli import build_parser

    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "dataset:\n  name: synthetic\n  memories_per_user: 10\n"
        "backend:\n  name: memmachine\n  base_url: http://core:8081\n" + backend_extra
    )
    return build_parser().parse_args(
        ["run", "--config", str(cfg), "--scenario", "search-load",
         "--users", "4", "--duration", "1", *flags]
    )


def test_a_shared_project_refuses_to_delete_on_exit_across_shards(monkeypatch, tmp_path):
    """Each shard tears down its own users, which assumes a project per user.
    With one shared project the first shard to finish would delete it under the
    others, so the combination is refused before the run starts.

    Args come from the real parser rather than a hand-built Namespace, so the
    test cannot drift as flags are added."""
    from ltm100.cli import _run

    launched = _stub_launch(monkeypatch)
    shared = "  project_id: shared\n  filter_by_producer: true\n"
    with pytest.raises(ValueError, match="cannot delete on exit"):
        _run(_run_args(tmp_path, shared, "--procs", "4"))
    assert launched == []
    # --no-delete-on-exit is the documented way through: the run must get as far
    # as launching all four shards.
    with pytest.raises(_Reached):
        _run(_run_args(tmp_path, shared, "--procs", "4", "--no-delete-on-exit"))
    assert launched == [4]


def test_a_shared_project_without_a_producer_filter_warns(monkeypatch, tmp_path, caplog):
    from ltm100.cli import _run

    _stub_launch(monkeypatch)
    with caplog.at_level("WARNING", logger="ltm100.cli"), pytest.raises(_Reached):
        _run(_run_args(tmp_path, "  project_id: shared\n"))
    assert "filter_by_producer" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="ltm100.cli"), pytest.raises(_Reached):
        _run(_run_args(tmp_path, "  project_id: shared\n  filter_by_producer: true\n"))
    assert "filter_by_producer" not in caplog.text


def test_a_quoted_user_id_is_refused_before_anything_is_sent():
    """The filter grammar cannot escape a quote, so the producer scope would break."""
    c = _mm(project_id="shared", filter_by_producer=True)
    seen = _capture(c)
    with pytest.raises(ValueError, match="quote"):
        asyncio.run(c.setup(["ok_user", "o'brien"]))
    assert seen == []
