"""Load core: orchestrates virtual users and drives a backend.

This is the closed-model runner: a fixed number of virtual users, each an
asyncio task that loops over its scenario plan emitting one request at a time
(in-flight = 1 per user). An optional global semaphore caps total concurrency
independently of the user count.

Termination is either time-based (duration) or count-based (total ops). The
runner drains in-flight requests at termination, then returns all recorded
OpResults.

Open-model (arrival-rate driven) behavior shares this same runner: arriving
sessions each consume a bounded number of ops from the Scenario plan (the
op mix is owned by the scenario, not a runner-level weight). Inter-arrival
is a Poisson process; congestion policy (rejection under overload) is
enforced by a bounded queue on the global concurrency cap and recorded as
status="rejected".
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from ltm100.common import DatasetAdapter, LTMClient, UserId
from ltm100.core.config import RunConfig
from ltm100.core.op import Op, OpResult, OpType, Scenario
from ltm100.metrics.recorder import InMemoryRecorder, MetricsRecorder

logger = logging.getLogger(__name__)


class LoadRunner:
    """Drives N virtual users through a Scenario against an LTMClient."""

    def __init__(
        self,
        *,
        client: LTMClient,
        dataset: DatasetAdapter,
        scenario: Scenario,
        config: RunConfig,
        recorder: MetricsRecorder | None = None,
    ) -> None:
        self.client = client
        self.dataset = dataset
        self.scenario = scenario
        self.config = config
        self.recorder = recorder or InMemoryRecorder()
        self._global_sem: asyncio.Semaphore | None = None
        self._admission_lock = asyncio.Lock()
        self._admitted = 0
        self._stop = asyncio.Event()
        self._ops_done = 0
        self._ops_lock = asyncio.Lock()
        self._start_time = 0.0

    async def run(self) -> list[OpResult]:
        users = self.shard_users(
            self.dataset.users(self.config.users, seed=self.config.seed)
        )
        self.users = users

        # Let the scenario reject a misconfigured dataset loudly, before any
        # setup/provisioning or user runs. A plan-time raise would be swallowed
        # by the gather(return_exceptions=True) in the user loops.
        validate = getattr(self.scenario, "validate", None)
        if validate is not None:
            validate(self.dataset)

        await self.client.setup(users)

        if self.config.global_concurrency > 0:
            self._global_sem = asyncio.Semaphore(self.config.global_concurrency)

        # Optional pre-ingest: fill each user's memory before measuring so
        # search scenarios run against populated memory. Excluded from metrics.
        if self.config.preingest:
            await self._preingest(users)

        self._start_time = time.monotonic()
        # deadline is None for pure count-based closed runs (no time bound); the
        # open model always has a duration (validated in RunConfig).
        deadline = (
            self._start_time + self.config.duration
            if self.config.duration > 0
            else None
        )

        if self.config.model == "open":
            assert deadline is not None  # validated by RunConfig
            await self._open_loop(users, deadline)
        else:
            await self._closed_loop(users, deadline)

        return self.recorder.raw()

    def shard_users(self, users: list[UserId]) -> list[UserId]:
        """This process's slice of the virtual users.

        Round-robin rather than contiguous blocks, so an ordered dataset does
        not hand one shard all the large conversations."""
        if self.config.procs == 1:
            return users
        return users[self.config.proc_index :: self.config.procs]

    async def _closed_loop(self, users: list[UserId], deadline: float) -> None:
        tasks = []
        for i, user in enumerate(users):
            # Staggered start for ramp-up: user i starts at i*ramp_step.
            start_delay = self._ramp_delay(i, len(users))
            tasks.append(
                asyncio.create_task(self._user_loop(user, start_delay, deadline))
            )
        if deadline is not None:
            asyncio.create_task(self._timer(deadline))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _open_loop(self, users: list[UserId], deadline: float) -> None:
        """Open model: a Poisson arrival process spawns user sessions, each
        running a bounded number of ops then completing. Concurrency is
        emergent (a function of arrival rate vs. service rate).

        The fixed pool of `users` provides tenant identities; arriving sessions
        draw users round-robin so per-user state already exists. Each arriving
        session is a coroutine; the loop keeps spawning until the deadline."""
        rng = random.Random(self.config.seed)
        rate = self.config.arrival_rate
        session_tasks: list[asyncio.Task] = []
        next_user = 0

        # Always run the timer to honor the deadline even if all sessions are
        # short; the arrival generator stops at the deadline.
        asyncio.create_task(self._timer(deadline))

        t = self._start_time
        while not self._should_stop():
            # Inter-arrival ~ Exponential(rate).
            gap = rng.expovariate(rate)
            t += gap
            now = time.monotonic()
            if t > deadline:
                break
            wait = max(0.0, t - now)
            if wait > 0:
                await asyncio.sleep(wait)
            if self._should_stop():
                break
            user = users[next_user % len(users)]
            next_user += 1
            session_tasks.append(
                asyncio.create_task(self._open_session(user, deadline))
            )

        # Let in-flight sessions finish (bounded by session_ops, so finite).
        if session_tasks:
            await asyncio.gather(*session_tasks, return_exceptions=True)

    async def _open_session(
        self, user: UserId, deadline: float
    ) -> None:
        """One arriving user's session: a bounded number of ops then exit.

        The op mix comes from the scenario's plan (same interface as the
        closed model); we consume up to `session_ops` ops from it. Each op
        acquires a global slot with a bounded queue; if the queue is full the
        op is rejected (status='rejected') rather than executed."""
        n_ops = self.config.session_ops
        rng_state = {"seed": self.config.seed, "user": user}
        plan = self.scenario.plan(user, self.dataset, rng_state)
        for _ in range(n_ops):
            if self._should_stop() or time.monotonic() >= deadline:
                return
            try:
                op = next(plan)
            except StopIteration:
                return
            if op.delay > 0:
                await asyncio.sleep(min(op.delay, self._remaining_until(deadline)))
                if self._should_stop() or time.monotonic() >= deadline:
                    return
            acquired = await self._acquire_slot_bounded()
            if not acquired:
                await self._record_rejected(op, user)
                continue
            try:
                await self._execute(user, op)
            finally:
                self._release_slot()

    async def _acquire_slot_bounded(self) -> bool:
        """Try to take a global concurrency slot, queuing up to queue_bound.

        Returns True if a slot was acquired, False if rejected (queue full).
        When there is no global cap, always succeeds."""
        if self._global_sem is None:
            return True
        # Reserve admission before waiting on the semaphore. The counter covers
        # both running and queued requests, unlike Semaphore._value, which can
        # only describe permits already taken. The lock makes the capacity
        # check and reservation one atomic decision across arriving sessions.
        capacity = self.config.global_concurrency + self.config.queue_bound
        async with self._admission_lock:
            if self._admitted >= capacity:
                return False
            self._admitted += 1

        try:
            await self._global_sem.acquire()
            return True
        except BaseException:
            # A cancelled waiter must return its admission reservation or the
            # queue would appear permanently full.
            self._admitted -= 1
            raise

    async def _record_rejected(self, op: Op, user: UserId) -> None:
        now = time.time()
        result = OpResult(
            type=op.type,
            user_id=user,
            started_at=now,
            ended_at=now,
            status="rejected",
            error_kind="queue_full",
            n_items=0,
        )
        await self.recorder.record(result)

    def _release_slot(self) -> None:
        if self._global_sem is not None:
            self._global_sem.release()
            self._admitted -= 1

    async def _preingest(self, users: list[UserId]) -> None:
        """Ingest a fraction of each user's memory stream, concurrently across
        users, with the global concurrency cap applied. Not recorded."""
        frac = max(0.0, min(self.config.preingest_fraction, 1.0))

        async def ingest_one(user: UserId) -> None:
            items = list(self.dataset.memory_stream(user))
            if frac < 1.0:
                keep = max(1, int(round(frac * len(items))))
                items = items[:keep]
            batch = getattr(self.client, "add_batch_size", 50)
            for start in range(0, len(items), batch):
                await self.client.add(user, items[start : start + batch])

        sem = self._global_sem

        async def guarded(user: UserId) -> None:
            if sem is not None:
                async with sem:
                    await ingest_one(user)
            else:
                await ingest_one(user)

        await asyncio.gather(*[guarded(u) for u in users], return_exceptions=True)

    def _ramp_delay(self, index: int, total: int) -> float:
        if self.config.rampup <= 0 or total <= 1:
            return 0.0
        step = self.config.rampup / total
        return step * index

    async def _timer(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self._stop.set()

    def _should_stop(self) -> bool:
        return self._stop.is_set()

    async def _reserve_op(self) -> bool:
        """Reserve budget for one op before emitting it (count-based).

        A successful reservation is a promise to run the op, so count-based
        termination records exactly `ops` results."""
        if self.config.ops <= 0:
            return True
        async with self._ops_lock:
            if self._ops_done >= self.config.ops:
                return False
            self._ops_done += 1
            return True

    async def _user_loop(
        self, user: UserId, start_delay: float, deadline: float | None
    ) -> None:
        if start_delay > 0:
            await asyncio.sleep(start_delay)

        rng_state = {"seed": self.config.seed, "user": user}
        plan = self.scenario.plan(user, self.dataset, rng_state)

        for op in plan:
            if self._should_stop():
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            if not await self._reserve_op():
                self._stop.set()
                return
            # A successful reservation is a promise to run this op, so we
            # don't check _should_stop() here: count-based termination is
            # exact (exactly `ops` results are recorded).
            if op.delay > 0:
                await asyncio.sleep(min(op.delay, self._remaining_until(deadline)))

            async with self._maybe_global_slot():
                await self._execute(user, op)

    def _remaining_until(self, deadline: float | None) -> float:
        if deadline is None:
            return 1e9  # effectively unbounded
        return max(deadline - time.monotonic(), 0.0)

    def _maybe_global_slot(self):
        if self._global_sem is None:
            class _Null:
                async def __aenter__(self_inner):
                    return self_inner

                async def __aexit__(self_inner, *a):
                    return False

            return _Null()
        return self._global_sem

    async def _execute(self, user: UserId, op: Op) -> None:
        started = time.time()
        status = "ok"
        error_kind = ""
        n_items = 0
        try:
            if op.type is OpType.ADD:
                uids = await self.client.add(user, op.items)
                n_items = len(uids)
            else:
                results = await self.client.search(user, op.query)
                n_items = len(results)
        except Exception as e:  # noqa: BLE001 - record any failure as op error
            status = "error"
            error_kind = type(e).__name__
            logger.debug("op %s for %s failed: %s", op.type.value, user, e)
        ended = time.time()
        result = OpResult(
            type=op.type,
            user_id=user,
            started_at=started,
            ended_at=ended,
            status=status,
            error_kind=error_kind,
            n_items=n_items,
        )
        await self.recorder.record(result)


__all__ = ["LoadRunner"]
