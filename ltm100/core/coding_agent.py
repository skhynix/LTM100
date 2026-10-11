"""Replay recorded agent activity under synthetic hook/MCP memory policies.

Source actions are data, never executed. LTM results do not change the source
trajectory. Backend latency does affect pacing because requests are awaited.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from typing import Any

from ltm100.agent import AgentMessage, AgentTask
from ltm100.common import DatasetAdapter, MemoryItem, QueryItem, UserId
from ltm100.core.agent_profile import AgentProfile
from ltm100.core.config import RunConfig
from ltm100.core.op import Op, OpType, Wait
from ltm100.text import split_chunks


def _content(message: AgentMessage, final_response: str | None = None) -> str:
    parts = [message.content] if message.content else []
    for call in message.tool_calls:
        if call.name == "finish":
            if final_response and final_response not in parts:
                parts.append(final_response)
        else:
            arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            parts.append(f"Tool call: {call.name}\nArguments: {arguments}")
    if message.role == "tool":
        parts.insert(0, f"Tool result: {message.name}")
    return "\n".join(parts)


def _rng(
    seed: int, user: str, occurrence: int, task: str, domain: str
) -> random.Random:
    key = json.dumps([seed, user, occurrence, task, domain], ensure_ascii=False)
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))


class CodingAgent:
    """One closed user replays tasks; one open arrival consumes exactly one task."""

    name = "coding-agent"
    strict_plans = True

    def __init__(
        self,
        profile: AgentProfile | None = None,
        top_k: int = 20,
        expand_context: int = 0,
        filter: str = "",
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(expand_context, bool)
            or not isinstance(expand_context, int)
            or expand_context < 0
        ):
            raise ValueError("expand_context must be a non-negative integer")
        self.profile = profile or AgentProfile()
        self.top_k = top_k
        self.expand_context = expand_context
        self.filter = filter
        self._count_run = False
        self.reset_phase()

    def validate_run(self, dataset: DatasetAdapter, users: list[UserId], *, model: str):
        if not callable(getattr(dataset, "task_stream", None)):
            raise TypeError("coding-agent requires a dataset with task_stream")
        if not users:
            raise ValueError("coding-agent requires at least one assigned user")
        # Do not retain all normalized trajectories. Keep only one lookahead
        # per user; the dataset validates source records before assigning users.
        for user in users:
            if not self._peek(user, dataset):
                raise ValueError(f"coding-agent has no tasks for user {user!r}")
            if (
                self._count_run
                and self.profile.integration == "mcp"
                and not any(self._eligible(task) for task in dataset.task_stream(user))
            ):
                raise ValueError(
                    f"coding-agent MCP policy cannot emit requests for user {user!r}"
                )

    def validate_config(self, config: RunConfig) -> None:
        self._count_run = (
            config.model == "closed" and config.ops > 0 and config.duration <= 0
        )
        if not self.profile.task_repeat and self._count_run:
            raise ValueError("task_repeat=false requires a duration-based run")

    def _eligible(self, task: AgentTask) -> bool:
        for index, message in enumerate(task.messages):
            if (
                message.role == "tool"
                and message.name != "finish"
                and self.profile.add_probability > 0
            ):
                return True
            terminal = index == len(task.messages) - 1 and any(
                call.name == "finish" for call in message.tool_calls
            )
            if (
                message.role == "assistant"
                and not terminal
                and self.profile.search_probability > 0
            ):
                return True
        return False

    def reset_phase(self) -> None:
        """Warm-up and measurement start at independent task/policy cursors."""
        self._streams: dict[UserId, Iterator[AgentTask]] = {}
        self._pending: dict[UserId, AgentTask | None] = {}
        self._occurrences: dict[UserId, int] = {}

    def _peek(self, user: UserId, dataset: DatasetAdapter) -> AgentTask | None:
        if user not in self._pending:
            stream = self._streams.setdefault(user, iter(dataset.task_stream(user)))
            task = next(stream, None)
            if task is None and self.profile.task_repeat:
                stream = self._streams[user] = iter(dataset.task_stream(user))
                task = next(stream, None)
            self._pending[user] = task
        return self._pending[user]

    def session_available(self, user: UserId, dataset: DatasetAdapter) -> bool:
        return self._peek(user, dataset) is not None

    def prepare_session(self, user: UserId, dataset: DatasetAdapter) -> dict[str, Any]:
        """Called synchronously at admission, before any session can finish."""
        task = self._peek(user, dataset)
        if task is None:
            raise RuntimeError(
                "cannot admit a session after its task pool is exhausted"
            )
        self._pending.pop(user)
        occurrence = self._occurrences.get(user, 0)
        self._occurrences[user] = occurrence + 1
        return {"agent_task": task, "task_occurrence": occurrence}

    def plan(
        self, user: UserId, dataset: DatasetAdapter, rng_state: dict[str, Any]
    ) -> Iterator[Op | Wait]:
        seed = rng_state.get("seed", 0)
        if "agent_task" in rng_state:
            yield from self._task_plan(
                user, rng_state["agent_task"], rng_state["task_occurrence"], seed
            )
            return
        while self.session_available(user, dataset):
            state = self.prepare_session(user, dataset)
            occurrence = state["task_occurrence"]
            if occurrence:
                timing = _rng(
                    seed, user, occurrence, state["agent_task"].trajectory_id, "gap"
                )
                yield Wait(
                    self.profile.delay(
                        timing, self.profile.task_gap, self.profile.task_gap_variation
                    )
                )
            yield from self._task_plan(user, state["agent_task"], occurrence, seed)

    def _task_plan(
        self, user: UserId, task: AgentTask, occurrence: int, seed: int
    ) -> Iterator[Op | Wait]:
        profile = self.profile
        hook = profile.integration in {"hook", "hybrid"}
        mcp = profile.integration in {"mcp", "hybrid"}
        timing = _rng(seed, user, occurrence, task.trajectory_id, "timing")
        policy = _rng(seed, user, occurrence, task.trajectory_id, "policy")
        initial = ""
        known_input = ""
        first_user = True
        calls: dict[str, tuple[str, str]] = {}
        for step, message in enumerate(task.messages):
            if message.role == "system":
                continue
            terminal = step == len(task.messages) - 1 and any(
                c.name == "finish" for c in message.tool_calls
            )
            text = _content(message, task.final_response if terminal else None)
            context = {
                "trajectory_id": task.trajectory_id,
                "instance_id": task.instance_id,
                "task_occurrence": occurrence,
                "step": step,
                "source_exit_status": task.exit_status,
                "source_resolved": task.resolved,
            }
            if message.role in {"user", "tool"}:
                if message.role == "tool":
                    _, name = calls[message.tool_call_id]
                    if name != "finish":
                        yield Wait(
                            profile.delay(
                                timing,
                                profile.tool_times.get(name, profile.tool_time),
                                profile.tool_time_variation,
                            )
                        )
                if message.role == "user" and first_user:
                    initial = text
                known_input = text
                include = hook and (
                    profile.hook_scope == "all-steps"
                    or (message.role == "user" and first_user)
                )
                if include:
                    yield self._search(initial, known_input, context, "hook")
                    yield from self._adds(user, text, message.role, context, "hook")
                if message.role == "user":
                    first_user = False
                if (
                    message.role == "tool"
                    and mcp
                    and message.name != "finish"
                    and policy.random() < profile.add_probability
                ):
                    action, _ = calls[message.tool_call_id]
                    yield from self._adds(
                        user, f"{action}\n{text}", "tool", context, "mcp"
                    )
            elif message.role == "assistant":
                if (
                    not terminal
                    and mcp
                    and policy.random() < profile.search_probability
                ):
                    yield self._search(initial, known_input, context, "mcp")
                yield Wait(
                    profile.delay(
                        timing,
                        profile.agent_step_time,
                        profile.agent_step_time_variation,
                    )
                )
                if hook and (profile.hook_scope == "all-steps" or terminal):
                    yield from self._adds(user, text, "assistant", context, "hook")
                for call in message.tool_calls:
                    calls[call.id] = (text, call.name)
        # Even a zero-delay task with no selected LTM calls must cooperate
        # with arrivals, deadlines and other users rather than busy-loop.
        yield Wait()

    def _search(self, initial: str, latest: str, context: dict, origin: str) -> Op:
        text = latest if not initial or latest == initial else f"{initial}\n\n{latest}"
        return Op(
            OpType.SEARCH,
            query=QueryItem(
                text[: self.profile.query_chars],
                self.top_k,
                self.expand_context,
                self.filter,
            ),
            context={**context, "origin": origin},
        )

    def _adds(
        self, user: str, text: str, role: str, context: dict, origin: str
    ) -> Iterator[Op]:
        for index, chunk in enumerate(split_chunks(text, self.profile.chunk_chars)):
            if chunk:
                yield Op(
                    OpType.ADD,
                    items=[MemoryItem(content=chunk, producer=user, role=role)],
                    context={**context, "origin": origin, "chunk_index": index},
                )


__all__ = ["CodingAgent"]
