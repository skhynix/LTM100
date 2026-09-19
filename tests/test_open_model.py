"""Tests for the open-model load path.

Open model: a Poisson arrival process spawns user sessions; concurrency is
emergent. A global concurrency cap + queue bound yields rejections under
overload, recorded as status='rejected'.
"""

from __future__ import annotations

import asyncio

import pytest

from ltm100.common import MemoryItem, ResultItem, UserId
from ltm100.core.config import RunConfig
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import Mixed


class FakeDataset:
    name = "fake"

    def __init__(self, n_memories: int = 20) -> None:
        self.n_memories = n_memories

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        return [f"u{i}" for i in range(n_users)]

    def memory_stream(self, user: UserId):
        for i in range(self.n_memories):
            yield MemoryItem(content=f"{user}-mem-{i}", producer=user)


class SlowBackend:
    """Backend whose add/search take a fixed delay to create congestion."""

    name = "slow"

    def __init__(self, delay: float = 0.1) -> None:
        self.delay = delay
        self.calls = 0
        self.searches: list[tuple[str, str]] = []  # (user, query string)

    async def setup(self, users):
        return

    async def add(self, user, items):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return [f"{user}-{i}" for i in range(len(items))]

    async def search(self, user, query):
        self.calls += 1
        await asyncio.sleep(self.delay)
        self.searches.append((user, query.query))
        return [ResultItem(content="hit")]

    async def teardown(self, users, *, delete):
        return


@pytest.mark.asyncio
async def test_open_model_runs_within_duration():
    ds = FakeDataset(n_memories=5)
    backend = SlowBackend(delay=0.01)
    cfg = RunConfig(
        users=3,
        duration=2.0,
        seed=0,
        model="open",
        arrival_rate=20.0,
        session_ops=4,
    )
    scenario = Mixed(search_weight=0.5, think=0.0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=scenario, config=cfg)
    import time

    t0 = time.monotonic()
    await runner.run()
    elapsed = time.monotonic() - t0
    assert elapsed < 4.0  # roughly bounded by duration + drain
    summary = runner.recorder.summary()
    assert summary["total"] > 0


@pytest.mark.asyncio
async def test_open_model_mixes_add_and_search():
    ds = FakeDataset(n_memories=10)
    backend = SlowBackend(delay=0.005)
    cfg = RunConfig(
        users=2,
        duration=2.0,
        seed=0,
        model="open",
        arrival_rate=30.0,
        session_ops=40,
    )
    scenario = Mixed(search_weight=0.8, think=0.0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=scenario, config=cfg)
    await runner.run()
    summary = runner.recorder.summary()
    assert "search" in summary["by_op"]
    assert "add" in summary["by_op"]
    # Search queries are content-derived from the user's own memories.
    for user, q in backend.searches:
        assert q.startswith(f"{user}-mem-")


@pytest.mark.asyncio
async def test_open_model_rejects_under_overload():
    """High arrival rate + slow backend + tiny queue bound => rejections."""
    ds = FakeDataset(n_memories=5)
    backend = SlowBackend(delay=0.2)  # slow
    cfg = RunConfig(
        users=4,
        duration=2.0,
        seed=0,
        model="open",
        arrival_rate=200.0,  # far exceeds service rate
        session_ops=10,
        global_concurrency=2,
        queue_bound=0,  # reject immediately when cap saturated
    )
    scenario = Mixed(search_weight=1.0, think=0.0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=scenario, config=cfg)
    await runner.run()
    statuses = [r.status for r in runner.recorder.raw()]
    assert "rejected" in statuses
    assert statuses.count("rejected") > 0


@pytest.mark.asyncio
async def test_open_model_requires_valid_params():
    with pytest.raises(ValueError):
        RunConfig(
            users=1, duration=1.0, model="open", session_ops=4  # arrival_rate=0
        )
    with pytest.raises(ValueError):
        RunConfig(
            users=1,
            ops=10,
            model="open",  # duration=0 not allowed for open
            arrival_rate=5.0,
            session_ops=4,
        )


@pytest.mark.asyncio
async def test_open_model_rejections_recorded_with_kind():
    ds = FakeDataset(n_memories=5)
    backend = SlowBackend(delay=0.2)
    cfg = RunConfig(
        users=2,
        duration=1.5,
        seed=0,
        model="open",
        arrival_rate=200.0,
        session_ops=8,
        global_concurrency=1,
        queue_bound=0,
    )
    scenario = Mixed(search_weight=1.0, think=0.0)
    runner = LoadRunner(client=backend, dataset=ds, scenario=scenario, config=cfg)
    await runner.run()
    rejected = [r for r in runner.recorder.raw() if r.status == "rejected"]
    assert all(r.error_kind == "queue_full" for r in rejected)
    # Rejected ops have zero latency.
    assert all(r.ended_at == r.started_at for r in rejected)


@pytest.mark.asyncio
async def test_queue_bound_limits_waiters_exactly():
    """C=1, Q=1 admits one runner and one waiter; the next request rejects."""
    cfg = RunConfig(
        users=1,
        duration=1.0,
        model="open",
        arrival_rate=1.0,
        session_ops=1,
        global_concurrency=1,
        queue_bound=1,
    )
    runner = LoadRunner(
        client=SlowBackend(), dataset=FakeDataset(), scenario=Mixed(), config=cfg
    )
    runner._global_sem = asyncio.Semaphore(cfg.global_concurrency)

    assert await runner._acquire_slot_bounded() is True  # running
    waiter = asyncio.create_task(runner._acquire_slot_bounded())
    await asyncio.sleep(0)  # waiter reserves the one queue position
    assert runner._admitted == 2

    assert await runner._acquire_slot_bounded() is False
    assert runner._admitted == 2

    runner._release_slot()
    assert await asyncio.wait_for(waiter, timeout=0.1) is True
    runner._release_slot()
    assert runner._admitted == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_returns_its_queue_position():
    cfg = RunConfig(
        users=1,
        duration=1.0,
        model="open",
        arrival_rate=1.0,
        session_ops=1,
        global_concurrency=1,
        queue_bound=1,
    )
    runner = LoadRunner(
        client=SlowBackend(), dataset=FakeDataset(), scenario=Mixed(), config=cfg
    )
    runner._global_sem = asyncio.Semaphore(cfg.global_concurrency)

    assert await runner._acquire_slot_bounded() is True
    waiter = asyncio.create_task(runner._acquire_slot_bounded())
    await asyncio.sleep(0)
    assert runner._admitted == 2

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert runner._admitted == 1

    runner._release_slot()
    assert runner._admitted == 0
