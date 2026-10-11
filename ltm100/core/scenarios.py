"""Scenarios for the closed and open load models.

Each Scenario turns a per-user dataset stream into a plan of Ops with delays.
The closed model loops back-to-back (delay=0) unless a scenario adds think
time; the open model consumes a bounded number of ops per arriving session
(see the runner's `_open_session`). The original scenarios wrap their
underlying streams to sustain load; coding-agent can also stop at task
exhaustion. The runner bounds request consumption via duration/ops (closed)
or session_ops (open). Request-free Wait events advance the agent timeline
without consuming those operation budgets.

Search queries are **content-derived**: built from the user's own
`memory_stream` items (one query per stored unit by default). `search-load`
can bound that pool to a fixed prefix for comparable memory-growth runs;
otherwise it is as large as the stream so cycling does not naively repeat a
single query (which would warm a server result cache and understate latency).
`chat-replay` instead derives each recall query from the upcoming user turn's
content. No scenario consumes a separate evaluation `query_stream`.

Scenarios here (coding-agent is implemented in its own module):
  - CodingAgent: recorded task replay under hook/MCP policies and simulated
                 agent/tool timing, preserving each open task boundary.
  - ChatReplay:   the primary workload — replay a chatbot-with-LTM integration
                  over the dataset's turn_stream (recall before a user turn,
                  then ingest the turn), with a configurable recall cadence.
                  (closed/open)
  - AddLoad:      infinite add stream over the user's memory_stream. (closed/open)
  - SearchLoad:   infinite content-derived search stream; assumes pre-ingest. (closed/open)
  - Mixed:        controllable add/search mixture (op mix via `search_weight`),
                  with think jitter. A simple, no-dialogue workload usable with
                  any dataset (incl. synthetic) — handy for quick congestion
                  probing. (closed/open)
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import replace
from itertools import islice
from typing import Any
from weakref import WeakKeyDictionary

from ltm100.common import DatasetAdapter, MemoryItem, QueryItem, Turn, UserId
from ltm100.core.chat_profile import ChatProfile, ChatSettings
from ltm100.core.coding_agent import CodingAgent
from ltm100.core.op import Op, OpType, Scenario

_MEMO: WeakKeyDictionary[DatasetAdapter, dict[UserId, list[MemoryItem]]] = (
    WeakKeyDictionary()
)


def _memories(dataset: DatasetAdapter, user: UserId) -> list[MemoryItem]:
    """Materialize a user's stream once per run, not once per session.

    The runner plans a session inside `_open_session`, on the event loop. Under
    `--model open` a session opens per arrival, so an uncached build costs
    `arrival_rate` x build-time per second of wall clock; past 1.0 the loop
    stops reading sockets and the run wedges (50k episodes build in ~0.47s, so
    --arrival-rate 20 demands 9.4x the loop). The stream is deterministic in
    (user, seed), so one build per user is equivalent. Cost is one corpus per
    live user, which the memory-scaling arms bound by running --users 1.
    """
    per_user = _MEMO.setdefault(dataset, {})
    cached = per_user.get(user)
    if cached is None:
        cached = per_user[user] = list(dataset.memory_stream(user))
    return cached


def _query_pool(
    scenario: Any, dataset: DatasetAdapter, user: UserId
) -> list[QueryItem]:
    """Cache the per-user query pool on the scenario instance.

    `_content_queries` builds one QueryItem per stored unit, so rebuilding it
    per session costs the same order as the corpus build itself (see
    `_memories`). The pool depends only on the user and the scenario's fixed
    top_k/expand_context/filter/query_limit. A bounded pool consumes only its
    prefix instead of materializing the complete corpus."""
    cache = scenario.__dict__.setdefault("_query_cache", {})
    pool = cache.get(user)
    if pool is None:
        limit = getattr(scenario, "query_limit", None)
        if limit is None:
            memories = _memories(dataset, user)
        else:
            memories = list(islice(dataset.memory_stream(user), limit))
            if len(memories) < limit:
                raise ValueError(
                    f"query_limit={limit} requires at least {limit} memory "
                    f"items for user {user!r}, but the dataset yielded "
                    f"{len(memories)}"
                )
        pool = cache[user] = _content_queries(
            memories,
            scenario.top_k,
            scenario.expand_context,
            scenario.filter,
        )
    return pool


def _seed_for(seed: int, user: str) -> int:
    """Combine the run seed and user id into a stable int seed.

    Avoids passing a str/tuple to random.Random, which Python 3.9+ warns
    about (hash-based seeding)."""
    h = (seed * 1_000_003) & 0xFFFFFFFF
    for ch in user.encode("utf-8"):
        h = (h ^ ch) * 1_000_003 & 0xFFFFFFFF
    return h


def _content_queries(
    memories: list[MemoryItem],
    top_k: int = 20,
    expand_context: int = 0,
    filter: str = "",
) -> list[QueryItem]:
    """Content-derived search queries, one per memory item.

    The pool is as large as the memory stream, so cycling it does not
    naively repeat a single query (which would warm a server result cache and
    understate latency). Each query is the memory item's own content."""
    return [
        QueryItem(
            query=m.content,
            top_k=top_k,
            expand_context=expand_context,
            filter=filter,
        )
        for m in memories
    ]


class AddLoad:
    """Pure storage throughput: each user adds its memory stream forever.

    The plan wraps the memory stream, so a duration run sustains add load
    until the runner stops it (count- or time-based)."""

    name = "add-load"

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        memories = _memories(dataset, user)
        if not memories:
            return
        i = 0
        while True:
            yield Op(type=OpType.ADD, items=[memories[i % len(memories)]], delay=0.0)
            i += 1


class SearchLoad:
    """Pure search throughput: each user searches content-derived queries forever.

    Assumes memories were pre-ingested (warm-up), so searches run against the
    user's own stored content. The query pool is built from the user's
    `memory_stream`; `query_limit` can keep it at an exact prefix size while
    the stored corpus grows. Each pass through the pool starts at a rotating
    offset so passes are not identical, and a small think time per op drifts
    users out of lockstep. `top_k` (default 20) sets the search depth. The plan
    is infinite; the runner bounds it via duration/ops (closed) or session_ops
    (open)."""

    name = "search-load"

    def __init__(
        self,
        top_k: int = 20,
        expand_context: int = 0,
        filter: str = "",
        query_limit: int | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if expand_context < 0:
            raise ValueError("expand_context must be >= 0")
        if query_limit is not None and query_limit <= 0:
            raise ValueError("query_limit must be > 0")
        self.top_k = top_k
        self.expand_context = expand_context
        self.filter = filter
        self.query_limit = query_limit

    def validate_run(
        self,
        dataset: DatasetAdapter,
        users: list[UserId],
        *,
        model: str,
    ) -> None:
        """Build bounded query pools before backend setup.

        A configured limit is an exact workload contract: silently using a
        smaller pool would make growth points incomparable. Validation also
        keeps a short dataset error out of the asynchronous user loops, where
        plan exceptions are otherwise collected as worker outcomes.
        """
        if self.query_limit is not None:
            for user in users:
                _query_pool(self, dataset, user)

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        rng = random.Random(_seed_for(rng_state.get("seed", 0), user))
        queries = _query_pool(self, dataset, user)
        n = len(queries)
        if n == 0:
            return
        stride = max(1, n // 7)  # coprime-ish rotation so passes differ
        pass_i = 0
        while True:
            offset = (pass_i * stride) % n
            for j in range(n):
                q = queries[(offset + j) % n]
                yield Op(type=OpType.SEARCH, query=q, delay=rng.uniform(0.0, 0.02))
            pass_i += 1


class Mixed:
    """A controllable add/search mixture over the user's `memory_stream`.

    A simple, no-dialogue workload: per op, a SEARCH is drawn with probability
    `search_weight`, else an ADD of one memory item. It needs no `turn_stream`,
    so it works with any dataset (including the synthetic one). Search queries
    are content-derived (from the user's `memory_stream`); adds reuse memories
    cyclically once the stream is exhausted (a returning user who has memories
    to re-add). A small think-time gap is attached as `Op.delay` so ops are not
    perfectly back-to-back. The plan is infinite; the runner bounds it via
    duration/ops (closed) or `session_ops` (open).

    The op mix is owned by this scenario (via `search_weight`), not by a
    runner-level weight, so the open and closed models share one Scenario
    interface. Handy as a quick congestion probe — a flat op stream with a
    tunable search/add ratio that runs under either load model.
    """

    name = "mixed"

    def __init__(
        self,
        search_weight: float = 0.8,
        think: float = 0.05,
        top_k: int = 20,
        expand_context: int = 0,
        filter: str = "",
    ) -> None:
        if not 0.0 <= search_weight <= 1.0:
            raise ValueError("search_weight must be in [0, 1]")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if expand_context < 0:
            raise ValueError("expand_context must be >= 0")
        self.search_weight = search_weight
        self.think = think
        self.top_k = top_k
        self.expand_context = expand_context
        self.filter = filter

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        seed = rng_state.get("seed", 0)
        rng = random.Random(_seed_for(seed, user))
        add_delay_rng = random.Random(_seed_for(seed, f"{user}:mixed-add-delay"))
        memories = _memories(dataset, user)
        if not memories:
            return
        queries = _query_pool(self, dataset, user)
        i_add = 0
        i_q = 0
        # Infinite: the runner bounds consumption per session / over duration.
        while True:
            is_search = rng.random() < self.search_weight
            if is_search:
                q = queries[i_q % len(queries)]
                i_q += 1
                # Keep the existing search RNG stream unchanged so enabling
                # add think time does not alter the seeded add/search mix.
                delay = rng.uniform(0.0, self.think)
                yield Op(type=OpType.SEARCH, query=q, delay=delay)
            else:
                item = memories[i_add % len(memories)]
                i_add += 1
                delay = add_delay_rng.uniform(0.0, self.think)
                yield Op(type=OpType.ADD, items=[item], delay=delay)


class ChatReplay:
    """Replay a chatbot-with-LTM workload over a multi-turn dialogue.

    Models the real integration pattern: a chatbot recalls relevant memories
    before answering a user, then ingests the conversation turn. The plan
    walks the dataset's structured `turn_stream` (user/assistant turns in
    order) and wraps it, so a duration run replays the conversation as many
    times as needed. For every turn:

      - if the turn is a **user** turn: issue a SEARCH whose query is the
        user turn's content (the recall step) when this turn's recall cadence
        fires (see `search_every`), then ADD the turn's items;
      - otherwise (assistant turn): just ADD the turn's items.

    Adds and the recall search are interleaved exactly as a live chatbot
    session would interleave them. The search query is derived from the
    upcoming user turn (not a separate evaluation question), so recall is
    driven by the conversation itself. `query_stream` is not used.

    Requires a dataset adapter that implements `turn_stream` (LongMemEval
    does); the run fails loudly at validation time otherwise. Profiled runs
    additionally use `session_stream` to preserve source conversation
    boundaries. Closed runs create `concurrent_sessions` fixed lanes per user;
    open runs admit arriving sessions into at most `max_sessions_per_user`
    lanes. Each lane remains sequential and shares the same backend user.
    Streams wrap until the runner stops them. A small `think` delay between ops
    mimics user/assistant think time so users drift out of lockstep.

    `search_every` (default 1) emits a recall search before every Nth user
    turn; 1 = recall before every user turn. The user-turn counter resets each
    replay pass, so each pass is an independent, reproducible chat session.

    **LLM answer time and user think time.** A real chatbot does not loop
    back-to-back: after recalling, the LLM spends time generating an answer,
    and the user spends time typing the next turn. These are modeled as two
    *mean* delays attached to the ops that follow them:

      - `answer_time`: an Exponential(mean=answer_time) delay is attached to
        the **first** ADD of the following assistant turn. Because `Op.delay`
        is applied before an op, this puts generation time after the user turn
        has been stored and before the assistant answer is stored.
      - `user_gap`: an Exponential(mean=user_gap) delay is attached to the
        **first** SEARCH (or first ADD if recall is skipped) of a user turn —
        the user is reading/typing while nothing happens at the LTM. Models
        user think/typing time before the next utterance.

    Both default to 0, which reproduces the original back-to-back loop. An
    optional `answer_time_variation` or `user_gap_variation` replaces the
    corresponding unbounded Exponential draw with a bounded Uniform draw at
    `mean * (1 +/- variation)`. A variation of 0 makes the delay fixed; 1
    allows the full 0-to-2x range. They apply uniformly unless a chat workload
    profile supplies group-specific values.
    `answer_time` only takes effect when there is a following assistant turn;
    `user_gap` only between turns (never before the very first turn of a
    replay pass, so each pass starts cleanly).

    `top_k` (default 20) is the recall search depth — how many memories the
    backend returns per recall. Larger values raise the retrieve/serialize cost
    of each search. A chat workload profile can override it per user group.
    """

    name = "chat-replay"

    def __init__(
        self,
        think: float = 0.05,
        search_every: int = 1,
        answer_time: float = 0.0,
        user_gap: float = 0.0,
        answer_time_variation: float | None = None,
        user_gap_variation: float | None = None,
        top_k: int = 20,
        expand_context: int = 0,
        filter: str = "",
        profile: ChatProfile | None = None,
    ) -> None:
        self.think = think
        self.search_every = max(1, int(search_every))
        if answer_time < 0:
            raise ValueError("answer_time must be >= 0")
        if user_gap < 0:
            raise ValueError("user_gap must be >= 0")
        self._validate_variation("answer_time_variation", answer_time_variation)
        self._validate_variation("user_gap_variation", user_gap_variation)
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if expand_context < 0:
            raise ValueError("expand_context must be >= 0")
        self.answer_time = answer_time
        self.user_gap = user_gap
        self.answer_time_variation = answer_time_variation
        self.user_gap_variation = user_gap_variation
        self.top_k = top_k
        self.expand_context = expand_context
        self.filter = filter
        self.profile = profile
        self._session_cache: dict[UserId, list[list[Turn]]] = {}

    def configure_users(self, users: list[UserId], *, seed: int) -> None:
        if self.profile is not None:
            self.profile.assign(users, seed=seed)

    def validate(self, dataset: DatasetAdapter) -> None:
        if not hasattr(dataset, "turn_stream"):
            raise ValueError(
                "chat-replay requires a dataset with a turn_stream "
                "(e.g. LongMemEval); the configured dataset does not provide one"
            )

    def validate_run(
        self,
        dataset: DatasetAdapter,
        users: list[UserId],
        *,
        model: str,
    ) -> None:
        self.validate(dataset)
        if self.profile is None:
            return
        if not hasattr(dataset, "session_stream"):
            raise ValueError(
                "profiled chat-replay requires a dataset with session_stream "
                "so conversation boundaries can be preserved"
            )
        if model == "open":
            missing_caps = [
                group.name
                for group in self.profile.groups
                if group.settings.max_sessions_per_user is None
            ]
            if missing_caps:
                names = ", ".join(missing_caps)
                raise ValueError(
                    "profiled chat-replay with the open load model requires "
                    "max_sessions_per_user for every group; missing for: "
                    f"{names}"
                )
        for user in users:
            settings = self._settings_for(user)
            required = (
                settings.max_sessions_per_user
                if model == "open"
                else settings.concurrent_sessions
            )
            assert required is not None  # validated above for profiled open runs
            available = len(self._sessions(user, dataset))
            if available < required:
                raise ValueError(
                    f"chat profile assigns {required} concurrent session(s) to "
                    f"user {user!r}, but the dataset provides only {available}"
                )

    def max_sessions_per_user(self, user: UserId) -> int | None:
        """Return the open-model session cap configured for ``user``."""
        if self.profile is None:
            return None
        return self._settings_for(user).max_sessions_per_user

    def session_group(self, user: UserId) -> str:
        """Return the reporting group for an arriving open-model session."""
        return self.profile.group_for(user).name if self.profile else ""

    def session_ids(self, user: UserId) -> list[int | None]:
        if self.profile is None:
            return [None]
        return list(range(self._settings_for(user).concurrent_sessions))

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        return self._plan(user, dataset, rng_state)

    def _plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op]:
        settings = self._settings_for(user)
        group = self.profile.group_for(user).name if self.profile else ""
        session_id = rng_state.get("session_id")
        seed_key = user if session_id is None else f"{user}:chat-session:{session_id}"
        rng = random.Random(_seed_for(rng_state.get("seed", 0), seed_key))

        if session_id is None:
            turns = list(dataset.turn_stream(user))
            if not turns:
                return
            while True:  # legacy flattened replay when no profile is active
                yield from self._conversation_ops(
                    turns, settings, rng, group=group, session_id=None
                )
        else:
            sessions = self._sessions(user, dataset)
            session_count = rng_state.get("session_count", settings.concurrent_sessions)
            assigned = sessions[session_id::session_count]
            if not assigned:
                return
            while True:
                for turns in assigned:
                    yield from self._conversation_ops(
                        turns,
                        settings,
                        rng,
                        group=group,
                        session_id=session_id,
                    )

    def _conversation_ops(
        self,
        turns: list[Turn],
        settings: ChatSettings,
        rng: random.Random,
        *,
        group: str,
        session_id: int | None,
    ) -> Iterator[Op]:
        """Replay one conversation in order within one concurrent lane."""
        user_turn_idx = 0
        previous_user_turn = False
        for turn_index, turn in enumerate(turns):
            role = turn.role.lower()
            is_user = role == "user"
            is_assistant = role == "assistant"
            turn_ops: list[Op] = []
            if is_user and turn.items:
                if user_turn_idx % settings.search_every == 0:
                    first_content = turn.items[0].content
                    turn_ops.append(
                        Op(
                            type=OpType.SEARCH,
                            query=QueryItem(
                                query=first_content,
                                top_k=settings.top_k,
                                expand_context=self.expand_context,
                                filter=self.filter,
                            ),
                            delay=rng.uniform(0.0, settings.think),
                            group=group,
                            session_id=session_id,
                        )
                    )
                user_turn_idx += 1
            for item in turn.items:
                item = item if item.role else replace(item, role=turn.role)
                turn_ops.append(
                    Op(
                        type=OpType.ADD,
                        items=[item],
                        delay=rng.uniform(0.0, settings.think),
                        group=group,
                        session_id=session_id,
                    )
                )
            if turn_ops:
                if is_user and turn_index > 0 and settings.user_gap > 0:
                    extra = self._timing_delay(
                        rng, settings.user_gap, settings.user_gap_variation
                    )
                    turn_ops[0] = replace(turn_ops[0], delay=turn_ops[0].delay + extra)
                if is_assistant and previous_user_turn and settings.answer_time > 0:
                    extra = self._timing_delay(
                        rng,
                        settings.answer_time,
                        settings.answer_time_variation,
                    )
                    turn_ops[0] = replace(turn_ops[0], delay=turn_ops[0].delay + extra)
                yield from turn_ops
            previous_user_turn = is_user and bool(turn_ops)

    def _sessions(self, user: UserId, dataset: DatasetAdapter) -> list[list[Turn]]:
        sessions = self._session_cache.get(user)
        if sessions is None:
            sessions = list(dataset.session_stream(user))
            self._session_cache[user] = sessions
        return sessions

    def _settings_for(self, user: UserId) -> ChatSettings:
        if self.profile is not None:
            return self.profile.group_for(user).settings
        return ChatSettings(
            think=self.think,
            search_every=self.search_every,
            answer_time=self.answer_time,
            user_gap=self.user_gap,
            top_k=self.top_k,
            answer_time_variation=self.answer_time_variation,
            user_gap_variation=self.user_gap_variation,
        )

    @staticmethod
    def _validate_variation(name: str, variation: float | None) -> None:
        if variation is not None and not 0.0 <= variation <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    @staticmethod
    def _timing_delay(
        rng: random.Random, mean: float, variation: float | None
    ) -> float:
        if variation is None:
            return rng.expovariate(1.0 / mean)
        return rng.uniform(mean * (1.0 - variation), mean * (1.0 + variation))


SCENARIOS: dict[str, type] = {
    CodingAgent.name: CodingAgent,
    ChatReplay.name: ChatReplay,
    AddLoad.name: AddLoad,
    SearchLoad.name: SearchLoad,
    Mixed.name: Mixed,
}


def get_scenario(name: str, **kwargs: Any) -> Scenario:
    cls = SCENARIOS.get(name)
    if cls is None:
        raise ValueError(f"unknown scenario: {name}")
    return cls(**kwargs)  # type: ignore[return-value]


__all__ = [
    "SCENARIOS",
    "AddLoad",
    "ChatReplay",
    "Mixed",
    "SearchLoad",
    "get_scenario",
]
