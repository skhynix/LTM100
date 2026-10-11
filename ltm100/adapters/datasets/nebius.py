"""Nebius SWE-rebench OpenHands trajectories for agent and corpus workloads.

The adapter retains source events without executing tools or choosing LTM
calls. Hugging Face/Parquet data stays disk-backed; records are normalized on
access rather than materializing the whole dataset as Python objects.
"""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ltm100.agent import AgentMessage, AgentTask, AgentToolCall
from ltm100.common import MemoryItem, UserId
from ltm100.text import split_chunks

DATASET_ID = "nebius/SWE-rebench-openhands-trajectories"
DEFAULT_REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"


class NebiusAdapter:
    """Load independent coding-agent execution attempts, including failures."""

    name = "nebius"

    def __init__(
        self,
        length: int | None = None,
        cache_dir: str | None = None,
        path: str | None = None,
        revision: str = DEFAULT_REVISION,
        chunk_chars: int = 3000,
    ) -> None:
        if length is not None and (
            isinstance(length, bool) or not isinstance(length, int) or length < 0
        ):
            raise ValueError("nebius length must be a non-negative integer or None")
        if (
            isinstance(chunk_chars, bool)
            or not isinstance(chunk_chars, int)
            or chunk_chars <= 0
        ):
            raise ValueError("nebius chunk_chars must be a positive integer")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("nebius revision must be a non-empty string")
        self.length = length
        self.cache_dir = cache_dir
        self.path = path
        self.revision = revision
        self.chunk_chars = chunk_chars
        self._records: Sequence[dict[str, Any]] | None = None
        self._assignments: dict[UserId, tuple[int, ...]] = {}
        self._validated = False

    # -- loading -----------------------------------------------------------

    def _load(self) -> Sequence[dict[str, Any]]:
        if self._records is not None:
            return self._records
        if self.path:
            source = Path(self.path).expanduser()
            if source.suffix.lower() == ".parquet":
                self._records = self._load_arrow(source)
            else:
                self._records = self._load_json(source)
        else:
            self._records = self._load_arrow()
        return self._records

    def _load_arrow(self, source: Path | None = None) -> Sequence[dict[str, Any]]:
        try:
            if source is None:
                from huggingface_hub import hf_hub_download

                source = Path(
                    hf_hub_download(
                        repo_id=DATASET_ID,
                        filename="trajectories.parquet",
                        repo_type="dataset",
                        revision=self.revision,
                        cache_dir=self.cache_dir,
                    )
                )
            return _ParquetRecords(source, self.length)
        except ImportError as exc:
            raise ImportError(
                'Nebius Hugging Face/Parquet loading requires pip install ".[datasets]"'
            ) from exc

    def _load_json(self, source: Path) -> list[dict[str, Any]]:
        if source.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("nebius path must be a .json, .jsonl, or .parquet file")
        records: list[dict[str, Any]] = []
        with source.open("rb") as stream:
            if source.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(stream, 1):
                    if self.length is not None and len(records) >= self.length:
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid nebius JSON at {source}:{line_number}"
                        ) from exc
                    records.append(record)
            else:
                import ijson

                try:
                    first = next(ijson.parse(stream), None)
                    if first is None or first[1] != "start_array":
                        raise ValueError(f"nebius JSON must contain an array: {source}")
                    stream.seek(0)
                    items = ijson.items(stream, "item", use_float=True)
                    while self.length is None or len(records) < self.length:
                        try:
                            records.append(next(items))
                        except StopIteration:
                            break
                except ijson.JSONError as exc:
                    raise ValueError(
                        f"nebius JSON must contain an array of valid records: {source}"
                    ) from exc
        return records

    def _task(self, index: int) -> AgentTask:
        try:
            return _normalize_task(self._load()[index])
        except (TypeError, ValueError, KeyError) as exc:
            source = self.path or f"{DATASET_ID}@{self.revision}"
            raise ValueError(
                f"invalid nebius trajectory at {source}, row {index}: {exc}"
            ) from exc

    # -- DatasetAdapter / AgentDataset -------------------------------------

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        """Shuffle selected tasks, then distribute them across virtual users.

        Each worker calls this with the whole run's user count before user
        sharding, so assignment is independent of process count. A short
        source repeats across users, whose backend identities remain unique.
        """
        if isinstance(n_users, bool) or not isinstance(n_users, int) or n_users < 0:
            raise ValueError("nebius n_users must be a non-negative integer")
        indices = list(range(len(self._load())))
        if n_users and not self._validated:
            # Generic scenarios materialize memory_stream inside async plans,
            # whose exceptions are collected by the runner. Validate selected
            # source records here, before backend setup, without retaining them.
            identifiers: set[str] = set()
            for index in indices:
                task = self._task(index)
                if task.trajectory_id in identifiers:
                    raise ValueError(
                        f"duplicate nebius trajectory_id: {task.trajectory_id}"
                    )
                identifiers.add(task.trajectory_id)
            self._validated = True
        random.Random(seed).shuffle(indices)
        self._assignments = {}
        if not indices:
            return []
        for i in range(n_users):
            user = f"neb_user_{i:05d}"
            assigned = (
                indices[i::n_users]
                if n_users <= len(indices)
                else [indices[i % len(indices)]]
            )
            self._assignments[user] = tuple(assigned)
        return list(self._assignments)

    def task_indices(self, user: UserId) -> tuple[int, ...]:
        """Selected-source row indices assigned to a user, in replay order."""
        try:
            return self._assignments[user]
        except KeyError as exc:
            raise ValueError(
                f"unknown nebius user {user!r}; call users() first"
            ) from exc

    def task_stream(self, user: UserId) -> Iterator[AgentTask]:
        """Yield assigned tasks once; the consuming scenario owns wrap-around."""
        for index in self.task_indices(user):
            yield self._task(index)

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        """Project source activity to a finite, backend-neutral corpus.

        Generic add/search scenarios can use this projection; it is not a
        coding-agent integration policy. Source events remain available in
        ``task_stream`` without chunking or loss of their boundaries.
        """
        for task in self.task_stream(user):
            for i, message in enumerate(task.messages):
                if message.role == "system":
                    continue
                terminal = i == len(task.messages) - 1
                content = _message_content(
                    message, task.final_response if terminal else None
                )
                for chunk in split_chunks(content, self.chunk_chars):
                    yield MemoryItem(content=chunk, producer=user, role=message.role)


def _message_content(message: AgentMessage, final_response: str | None) -> str:
    parts = [message.content] if message.content.strip() else []
    if message.role == "tool":
        parts.insert(0, f"Tool result: {message.name}")
    for call in message.tool_calls:
        if final_response is not None and call.name == "finish":
            if (
                final_response.strip()
                and final_response.strip() != message.content.strip()
            ):
                parts.append(final_response)
        else:
            arguments = json.dumps(
                call.arguments, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            parts.append(f"Tool call: {call.name}\nArguments: {arguments}")
    return "\n\n".join(parts)


class _ParquetRecords(Sequence):
    """Indexed source rows, caching one projected row group at a time.

    Keep the downloaded Parquet rather than expanding it to a second, much
    larger Arrow dataset. Exclude patches and tool definitions from decoding.
    """

    _columns = (
        "trajectory_id",
        "instance_id",
        "repo",
        "trajectory",
        "exit_status",
        "resolved",
    )

    def __init__(self, source: Path, length: int | None):
        from pyarrow import parquet

        self._file = parquet.ParquetFile(source)
        missing = set(self._columns) - set(self._file.schema_arrow.names)
        if missing:
            raise ValueError(f"nebius Parquet is missing columns: {sorted(missing)}")
        self._size = self._file.metadata.num_rows
        if length is not None:
            self._size = min(length, self._size)
        self._offsets = [0]
        for group in range(self._file.num_row_groups):
            self._offsets.append(
                self._offsets[-1] + self._file.metadata.row_group(group).num_rows
            )
        self._group = None
        self._table = None

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._size))]
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        group = bisect_right(self._offsets, index) - 1
        if group != self._group:
            # Release the previous group before decoding its replacement.
            self._table = None
            self._table = self._file.read_row_group(group, columns=list(self._columns))
            self._group = group
        return self._table.slice(index - self._offsets[group], 1).to_pylist()[0]


def _text(value: Any, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{field} must be a {'non-empty ' if nonempty else ''}string")
    return value


def _normalize_task(record: dict[str, Any]) -> AgentTask:
    if not isinstance(record, dict):
        raise TypeError("record must be an object")
    trajectory_id = _text(record["trajectory_id"], "trajectory_id", nonempty=True)
    instance_id = _text(record["instance_id"], "instance_id", nonempty=True)
    repo = _text(record["repo"], "repo", nonempty=True)
    exit_status = _text(record["exit_status"], "exit_status", nonempty=True)
    resolved = record["resolved"]
    if resolved not in (0, 1) or not isinstance(resolved, (bool, int)):
        raise ValueError("resolved must be 0 or 1")
    source_messages = record["trajectory"]
    if not isinstance(source_messages, list) or not source_messages:
        raise ValueError("trajectory must be a non-empty list")

    messages: list[AgentMessage] = []
    calls: dict[str, str] = {}
    observations: set[str] = set()
    for position, message in enumerate(source_messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {position} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {position} has an unsupported role: {role!r}")
        content = _text(message.get("content"), f"message {position} content")
        tool_calls: list[AgentToolCall] = []
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if raw_calls is not None and not isinstance(raw_calls, list):
                raise ValueError(f"message {position} tool_calls must be a list")
            for call in raw_calls or []:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError(f"message {position} has an invalid tool call")
                call_id = _text(call.get("id"), "tool call id", nonempty=True)
                if call_id in calls:
                    raise ValueError(f"duplicate tool call id: {call_id}")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise TypeError("tool call function must be an object")
                name = _text(function.get("name"), "tool name", nonempty=True)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                else:
                    # Return an independent snapshot even for already decoded input.
                    arguments = json.loads(json.dumps(arguments, allow_nan=False))
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must decode to an object")
                json.dumps(arguments, allow_nan=False)
                calls[call_id] = name
                tool_calls.append(AgentToolCall(call_id, name, arguments))

        call_id = None
        name = None
        if role == "tool":
            call_id = _text(message.get("tool_call_id"), "tool_call_id", nonempty=True)
            name = _text(message.get("name"), "tool observation name", nonempty=True)
            if calls.get(call_id) != name:
                raise ValueError(f"unmatched tool observation: {call_id} ({name})")
            if call_id in observations:
                raise ValueError(f"duplicate tool observation: {call_id}")
            observations.add(call_id)
        messages.append(AgentMessage(role, content, tuple(tool_calls), call_id, name))

    if not any(message.role == "user" for message in messages):
        raise ValueError("trajectory has no user request")
    if not any(message.role == "assistant" for message in messages):
        raise ValueError("trajectory has no assistant activity")
    # A terminal finish action has no observation. Interrupted attempts can
    # also end with unobserved calls; do not invent the missing results.
    trailing_calls = {call.id for call in messages[-1].tool_calls}
    for call_id, name in calls.items():
        if (
            call_id not in observations
            and name != "finish"
            and (exit_status == "submit" or call_id not in trailing_calls)
        ):
            raise ValueError(f"missing tool observation: {call_id} ({name})")

    final_response = None
    if messages[-1].role == "assistant":
        finish_calls = [c for c in messages[-1].tool_calls if c.name == "finish"]
        if len(finish_calls) > 1:
            raise ValueError("multiple terminal finish calls")
        if finish_calls:
            final_response = _text(
                finish_calls[0].arguments.get("message"), "finish message"
            )
    return AgentTask(
        trajectory_id,
        instance_id,
        repo,
        tuple(messages),
        exit_status,
        bool(resolved),
        final_response,
    )


__all__ = ["NebiusAdapter"]
