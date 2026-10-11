"""Load core: orchestrates virtual users and drives a backend.

This is the closed-model runner: a fixed number of virtual users, each with one
or more scenario lanes that emit one request at a time. Ordinary scenarios use
one lane per user; a profiled chat replay can use several independent session
lanes. An optional global semaphore caps total concurrency independently of
the user and lane counts.

Termination is either time-based (duration) or count-based (total ops). The
runner drains in-flight requests at termination, then returns all recorded
OpResults.

Open-model (arrival-rate driven) behavior shares this same runner: arriving
sessions each consume a bounded number of ops from the Scenario plan (the
op mix is owned by the scenario, not a runner-level weight). Inter-arrival
is a Poisson process. A scenario may cap active sessions per user; rejected
session arrivals are counted separately from request metrics. Request-level
congestion policy is enforced by a bounded queue on the global concurrency
cap and recorded as status="rejected".
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from itertools import islice
from typing import Any

from ltm100.common import DatasetAdapter, LTMClient, UserId
from ltm100.core.config import RunConfig
from ltm100.core.op import Op, OpResult, OpType, Scenario, Wait
from ltm100.metrics.recorder import InMemoryRecorder, MetricsRecorder

logger = logging.getLogger(__name__)


@dataclass
class SessionCounts:
    offered: int = 0
    admitted: int = 0
    rejected: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "offered": self.offered,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "rejection_rate": (self.rejected / self.offered if self.offered else 0.0),
        }


@dataclass
class SessionAdmissionStats(SessionCounts):
    """Open-model session arrivals, separate from request-level metrics."""

    by_group: dict[str, SessionCounts] | None = None

    def record(self, status: str, group: str = "") -> None:
        setattr(self, status, getattr(self, status) + 1)
        if group:
            if self.by_group is None:
                self.by_group = {}
            counts = self.by_group.setdefault(group, SessionCounts())
            setattr(counts, status, getattr(counts, status) + 1)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = super().as_dict()
        if self.by_group:
            result["by_group"] = {
                group: counts.as_dict()
                for group, counts in sorted(self.by_group.items())
            }
        return result


@dataclass
class PreingestStats:
    """Successfully submitted input items from the unmeasured pre-ingest."""

    users: int = 0
    input_items: int = 0
    min_items_per_user: int | None = None
    max_items_per_user: int | None = None

    def record(self, count: int) -> None:
        self.users += 1
        self.input_items += count
        if self.min_items_per_user is None or count < self.min_items_per_user:
            self.min_items_per_user = count
        if self.max_items_per_user is None or count > self.max_items_per_user:
            self.max_items_per_user = count

    def merge(self, other: PreingestStats) -> None:
        if other.users == 0:
            return
        assert other.min_items_per_user is not None
        assert other.max_items_per_user is not None
        self.users += other.users
        self.input_items += other.input_items
        if (
            self.min_items_per_user is None
            or other.min_items_per_user < self.min_items_per_user
        ):
            self.min_items_per_user = other.min_items_per_user
        if (
            self.max_items_per_user is None
            or other.max_items_per_user > self.max_items_per_user
        ):
            self.max_items_per_user = other.max_items_per_user

    def as_dict(self) -> dict[str, int | None]:
        return {
            "users": self.users,
            "input_items": self.input_items,
            "min_items_per_user": self.min_items_per_user,
            "max_items_per_user": self.max_items_per_user,
        }


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
        on_measure_start: Callable[[], Awaitable[None]] | None = None,
        on_measure_end: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.client = client
        self.dataset = dataset
        self.scenario = scenario
        self.config = config
        self.recorder = recorder or InMemoryRecorder()
        self.on_measure_start = on_measure_start
        self.on_measure_end = on_measure_end
        self._global_sem: asyncio.Semaphore | None = None
        self._admission_lock = asyncio.Lock()
        self._admitted = 0
        self._stop = asyncio.Event()
        self._ops_done = 0
        self._ops_lock = asyncio.Lock()
        self._start_time = 0.0
        self._record_results = True
        self.session_stats = SessionAdmissionStats()
        self.preingest_stats = PreingestStats()
        self.measurement_started_at: float | None = None
        self.measurement_ended_at: float | None = None

    async def run(self) -> list[OpResult]:
        all_users = self._configure_users(
            self.dataset.users(self.config.users, seed=self.config.seed)
        )
        users = self.shard_users(all_users)
        self.users = users

        validate_config = getattr(self.scenario, "validate_config", None)
        if validate_config is not None:
            validate_config(self.config)

        # Let the scenario reject a misconfigured dataset loudly, before any
        # setup/provisioning or user runs. A plan-time raise would be swallowed
        # by the gather(return_exceptions=True) in the user loops.
        validate_run = getattr(self.scenario, "validate_run", None)
        validate = getattr(self.scenario, "validate", None)
        if validate_run is not None:
            validate_run(self.dataset, users, model=self.config.model)
        elif validate is not None:
            validate(self.dataset)

        await self.client.setup(users)

        if self.config.global_concurrency > 0:
            self._global_sem = asyncio.Semaphore(self.config.global_concurrency)

        # Optional pre-ingest: fill each user's memory before measuring so
        # search scenarios run against populated memory. Excluded from metrics.
        if self.config.preingest:
            await self._preingest(users)

        # Warm-up drives the real workload against the real backend but does
        # not consume the measured duration/op budget and is never recorded.
        # Running it as a separate phase also drains in-flight work before the
        # measurement hooks fire, so client and server-side metric windows
        # share an exact boundary.
        if self.config.warmup > 0:
            self._record_results = False
            await self._run_workload(duration=self.config.warmup)
            self._reset_phase_state()

        # Optional observation hooks bracketing the measured window (e.g. the
        # CLI's server-metrics snapshots). Excluded: setup and pre-ingest
        # before, warm-up before, and teardown after (teardown lives in the
        # caller). A hook failure is logged and swallowed -- observation never
        # cancels the benchmark it observes.
        await self._call_hook(self.on_measure_start, "on_measure_start")

        self._record_results = True
        self.measurement_started_at = time.time()
        await self._run_workload(duration=self.config.duration)
        self.measurement_ended_at = time.time()

        await self._call_hook(self.on_measure_end, "on_measure_end")
        return self.recorder.raw()

    def _configure_users(self, users: list[UserId]) -> list[UserId]:
        """Resolve whole-run scenario state before process-local sharding."""
        configure = getattr(self.scenario, "configure_users", None)
        if configure is not None:
            configure(users, seed=self.config.seed)
        return users

    async def _run_workload(self, *, duration: float) -> None:
        """Run one workload phase.

        ``duration`` is always positive for warm-up. For the measured phase it
        may be zero when an exact operation count is the termination condition.
        """
        self._start_time = time.monotonic()
        reset_phase = getattr(self.scenario, "reset_phase", None)
        if reset_phase is not None:
            reset_phase()
        # deadline is None for pure count-based closed runs (no time bound); the
        # open model always has a duration (validated in RunConfig).
        deadline = self._start_time + duration if duration > 0 else None

        if self.config.model == "open":
            assert deadline is not None  # validated by RunConfig
            await self._open_loop(self.users, deadline)
        else:
            await self._closed_loop(self.users, deadline)

    def _reset_phase_state(self) -> None:
        """Reset termination state after warm-up, leaving backend state intact."""
        self._stop = asyncio.Event()
        self._ops_done = 0
        self._admitted = 0
        self.session_stats = SessionAdmissionStats()

    async def _call_hook(
        self, hook: Callable[[], Awaitable[None]] | None, what: str
    ) -> None:
        if hook is None:
            return
        try:
            await hook()
        except Exception:  # hooks observe the run, never break it
            logger.warning("%s hook failed", what, exc_info=True)

    def shard_users(self, users: list[UserId]) -> list[UserId]:
        """This process's slice of the virtual users.

        Round-robin rather than contiguous blocks, so an ordered dataset does
        not hand one shard all the large conversations."""
        if self.config.procs == 1:
            return users
        return users[self.config.proc_index :: self.config.procs]

    async def _closed_loop(self, users: list[UserId], deadline: float | None) -> None:
        tasks = []
        for i, user in enumerate(users):
            # Staggered start for ramp-up: user i starts at i*ramp_step.
            start_delay = self._ramp_delay(i, len(users))
            session_ids = self._session_ids(user)
            for session_id in session_ids:
                tasks.append(
                    asyncio.create_task(
                        self._user_loop(
                            user,
                            start_delay,
                            deadline,
                            session_id=session_id,
                        )
                    )
                )
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        self._check_plan_outcomes(outcomes)

    def _check_plan_outcomes(self, outcomes: list[Any]) -> None:
        """Opt-in strict plan failures; existing scenarios retain their behavior."""
        if getattr(self.scenario, "strict_plans", False):
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

    def _session_ids(self, user: UserId) -> list[int | None]:
        session_ids = getattr(self.scenario, "session_ids", None)
        return session_ids(user) if session_ids is not None else [None]

    async def _open_loop(self, users: list[UserId], deadline: float) -> None:
        """Open model: a Poisson arrival process spawns user sessions, each
        running a bounded number of ops then completing. Concurrency is
        emergent (a function of arrival rate vs. service rate).

        The fixed pool of `users` provides tenant identities; arriving sessions
        draw users round-robin so per-user state already exists. A scenario may
        impose a per-user session cap; a saturated user's arrival is rejected,
        never reassigned to another tenant. Each admitted session is a
        coroutine; the loop keeps generating arrivals until the deadline."""
        rng = random.Random(self.config.seed)
        rate = self.config.arrival_rate
        session_tasks: list[asyncio.Task] = []
        next_user = 0
        active_sessions: dict[UserId, set[int]] = {user: set() for user in users}

        try:
            t = self._start_time
            while not self._should_stop() and not self._deadline_reached(deadline):
                # Inter-arrival ~ Exponential(rate).
                gap = rng.expovariate(rate)
                t += gap
                now = time.monotonic()
                if t > deadline:
                    break
                wait = max(0.0, t - now)
                if not await self._wait_delay(wait, deadline) or self._should_stop():
                    break
                user = users[next_user % len(users)]
                next_user += 1
                available = getattr(self.scenario, "session_available", None)
                if available is not None and not available(user, self.dataset):
                    # Exhausted finite sources are not congestion rejections.
                    continue
                group_for = getattr(self.scenario, "session_group", None)
                group = group_for(user) if group_for is not None else ""
                if self._deadline_reached(deadline):
                    break
                if self._record_results:
                    self.session_stats.record("offered", group)
                admission = self._admit_open_session(user, active_sessions)
                if admission is None:
                    if self._record_results:
                        self.session_stats.record("rejected", group)
                    continue
                session_id, session_count = admission
                try:
                    prepare = getattr(self.scenario, "prepare_session", None)
                    session_state = prepare(user, self.dataset) if prepare else None
                    if self._record_results:
                        self.session_stats.record("admitted", group)
                    session_tasks.append(
                        asyncio.create_task(
                            self._open_session(
                                user,
                                deadline,
                                session_id=session_id,
                                session_count=session_count,
                                active_sessions=active_sessions,
                                session_state=session_state,
                            )
                        )
                    )

                except BaseException:
                    # No task owns this reservation if preparation fails.
                    if session_id is not None:
                        active_sessions[user].discard(session_id)
                    raise
        except BaseException:
            # Stop admissions and further dispatch before the caller tears down
            # backend state. Already executing requests are allowed to finish.
            self._stop.set()
            raise
        finally:
            # Preparation errors must not bypass session draining. Preserve the
            # original error; inspect plan outcomes only after a normal arrival loop.
            outcomes = await asyncio.gather(*session_tasks, return_exceptions=True)
        self._check_plan_outcomes(outcomes)

    def _admit_open_session(
        self,
        user: UserId,
        active: dict[UserId, set[int]],
    ) -> tuple[int | None, int | None] | None:
        """Reserve one profile-defined session lane for the selected user."""
        cap_for = getattr(self.scenario, "max_sessions_per_user", None)
        if cap_for is None:
            return None, None
        cap = cap_for(user)
        if cap is None:
            return None, None
        occupied = active[user]
        if len(occupied) >= cap:
            return None
        session_id = next(i for i in range(cap) if i not in occupied)
        occupied.add(session_id)
        return session_id, cap

    async def _open_session(
        self,
        user: UserId,
        deadline: float,
        *,
        session_id: int | None,
        session_count: int | None,
        active_sessions: dict[UserId, set[int]],
        session_state: dict[str, Any] | None = None,
    ) -> None:
        """One arriving user's session: a bounded number of ops then exit.

        The op mix comes from the scenario's plan (same interface as the
        closed model); we consume up to `session_ops` ops from it. Each op
        acquires a global slot with a bounded queue; if the queue is full the
        op is rejected (status='rejected') rather than executed."""
        n_ops = self.config.session_ops
        rng_state = {"seed": self.config.seed, "user": user}
        if session_id is not None:
            rng_state["session_id"] = session_id
            rng_state["session_count"] = session_count
        if session_state is not None:
            rng_state.update(session_state)
        try:
            plan = self.scenario.plan(user, self.dataset, rng_state)
            consumed = 0
            while consumed < n_ops:
                if self._should_stop() or self._deadline_reached(deadline):
                    return
                try:
                    op = next(plan)
                except StopIteration:
                    return
                if isinstance(op, Wait):
                    if not await self._wait_event(op, deadline):
                        return
                    continue
                consumed += 1
                if not await self._wait_delay(op.delay, deadline):
                    return
                if self._should_stop():
                    return
                acquired = await self._acquire_slot_bounded(deadline)
                if acquired is None:
                    return
                if not acquired:
                    if self._should_stop() or self._deadline_reached(deadline):
                        return
                    await self._record_rejected(op, user)
                    continue
                try:
                    if self._should_stop():
                        return
                    await self._execute(user, op)
                finally:
                    self._release_slot()
        finally:
            if session_id is not None:
                active_sessions[user].remove(session_id)

    async def _acquire_slot_bounded(self, deadline: float | None = None) -> bool | None:
        """Try to take a global concurrency slot, queuing up to queue_bound.

        Returns True if acquired, False for a full queue, or None if expired.
        Expiration is not a congestion rejection and must not enter metrics.
        """
        if self._deadline_reached(deadline):
            return None
        if self._global_sem is None:
            return True
        # Reserve admission before waiting on the semaphore. The counter covers
        # both running and queued requests, unlike Semaphore._value, which can
        # only describe permits already taken. The lock makes the capacity
        # check and reservation one atomic decision across arriving sessions.
        capacity = self.config.global_concurrency + self.config.queue_bound
        async with self._admission_lock:
            if self._deadline_reached(deadline):
                return None
            if self._admitted >= capacity:
                return False
            self._admitted += 1

        try:
            acquired = await self._acquire_global_slot(deadline)
        except BaseException:
            # A cancelled waiter must return its admission reservation or the
            # queue would appear permanently full.
            self._admitted -= 1
            raise
        if not acquired:
            self._admitted -= 1
            return None
        return True

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
            group=op.group,
            session_id=op.session_id,
            context=op.context,
        )
        if self._record_results:
            await self.recorder.record(result)

    def _release_slot(self) -> None:
        if self._global_sem is not None:
            self._global_sem.release()
            self._admitted -= 1

    async def _preingest(self, users: list[UserId]) -> None:
        """Pre-ingest each user's corpus outside the measured result stream.

        The legacy fraction mode materializes a user's corpus because the
        fraction depends on its final length. Exact-count mode consumes only
        the requested prefix in batches, which keeps client memory bounded for
        large memory-growth experiments.
        """
        frac = max(0.0, min(self.config.preingest_fraction, 1.0))
        exact = self.config.preingest_items_per_user

        async def ingest_one(user: UserId) -> int:
            batch = getattr(self.client, "add_batch_size", 50)
            if batch <= 0:
                raise ValueError(f"add_batch_size must be > 0, got {batch}")

            if exact is not None:
                stream = iter(self.dataset.memory_stream(user))
                submitted = 0
                while submitted < exact:
                    items = list(islice(stream, min(batch, exact - submitted)))
                    if not items:
                        raise ValueError(
                            f"preingest_items_per_user={exact} requires at least "
                            f"{exact} memory items for user {user!r}, but the "
                            f"dataset yielded {submitted}"
                        )
                    await self.client.add(user, items)
                    submitted += len(items)
                return submitted

            if frac == 0.0:
                return 0
            items = list(self.dataset.memory_stream(user))
            if frac < 1.0:
                keep = max(1, round(frac * len(items)))
                items = items[:keep]
            for start in range(0, len(items), batch):
                await self.client.add(user, items[start : start + batch])
            return len(items)

        sem = self._global_sem

        async def guarded(user: UserId) -> int:
            if sem is not None:
                async with sem:
                    return await ingest_one(user)
            return await ingest_one(user)

        outcomes = await asyncio.gather(
            *[guarded(u) for u in users], return_exceptions=True
        )
        failures = [
            (user, outcome)
            for user, outcome in zip(users, outcomes)
            if isinstance(outcome, BaseException)
        ]
        if failures:
            user, error = failures[0]
            raise RuntimeError(
                f"pre-ingest failed for {len(failures)} user(s); first failure "
                f"for {user}: {type(error).__name__}: {error}"
            ) from error
        for outcome in outcomes:
            assert isinstance(outcome, int)
            self.preingest_stats.record(outcome)

    def _ramp_delay(self, index: int, total: int) -> float:
        if self.config.rampup <= 0 or total <= 1:
            return 0.0
        step = self.config.rampup / total
        return step * index

    def _should_stop(self) -> bool:
        return self._stop.is_set()

    def _deadline_reached(self, deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    async def _wait_delay(self, delay: float, deadline: float | None) -> bool:
        """Wait only within the dispatch window; count-only delays are unbounded."""
        if self._deadline_reached(deadline):
            return False
        if delay > 0:
            await asyncio.sleep(
                delay
                if deadline is None
                else min(delay, self._remaining_until(deadline))
            )
        return not self._deadline_reached(deadline)

    async def _wait_event(self, event: Wait, deadline: float | None) -> bool:
        """Advance a non-request timeline, including cooperative zero waits."""
        if event.delay == 0:
            await asyncio.sleep(0)
        return await self._wait_delay(event.delay, deadline)

    async def _acquire_global_slot(self, deadline: float | None) -> bool:
        """Acquire before the deadline without cancelling an executing request."""
        if self._deadline_reached(deadline):
            return False
        sem = self._global_sem
        if sem is None:
            return True
        try:
            if deadline is not None and sem.locked():
                await asyncio.wait_for(
                    sem.acquire(), timeout=self._remaining_until(deadline)
                )
            else:
                # Avoid task creation on the uncontended workload hot path.
                await sem.acquire()
        except asyncio.TimeoutError:
            return False
        # Acquisition may resume after the timeout boundary. Return the permit
        # before reporting expiry; callers must not release it a second time.
        if self._deadline_reached(deadline):
            sem.release()
            return False
        return True

    async def _reserve_op(self) -> bool:
        """Reserve budget for one op before emitting it (count-based).

        With no duration limit, a reservation promises to run the op, so pure
        count-based termination records exactly `ops` results. A duration cap
        can expire a reserved but not yet dispatched operation."""
        if not self._record_results or self.config.ops <= 0:
            return True
        async with self._ops_lock:
            if self._ops_done >= self.config.ops:
                return False
            self._ops_done += 1
            return True

    async def _user_loop(
        self,
        user: UserId,
        start_delay: float,
        deadline: float | None,
        *,
        session_id: int | None = None,
    ) -> None:
        if not await self._wait_delay(start_delay, deadline):
            return

        rng_state = {"seed": self.config.seed, "user": user}
        if session_id is not None:
            rng_state["session_id"] = session_id
        plan = self.scenario.plan(user, self.dataset, rng_state)

        for op in plan:
            if self._should_stop():
                return
            if self._deadline_reached(deadline):
                return
            if isinstance(op, Wait):
                if (
                    self._record_results
                    and self.config.ops > 0
                    and self._ops_done >= self.config.ops
                ):
                    return
                if not await self._wait_event(op, deadline):
                    return
                continue
            if not await self._reserve_op():
                self._stop.set()
                return
            # A count stop cannot discard another lane's reserved operation.
            # Only an explicit duration deadline can expire pending dispatch.
            if not await self._wait_delay(op.delay, deadline):
                return
            if not await self._acquire_global_slot(deadline):
                return
            try:
                await self._execute(user, op)
            finally:
                if self._global_sem is not None:
                    self._global_sem.release()

    def _remaining_until(self, deadline: float | None) -> float:
        return (
            float("inf") if deadline is None else max(deadline - time.monotonic(), 0.0)
        )

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
            group=op.group,
            session_id=op.session_id,
            context=op.context,
        )
        if self._record_results:
            await self.recorder.record(result)


__all__ = ["LoadRunner"]
