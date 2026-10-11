"""LongMemEval dataset adapter.

LongMemEval (xiaowu0162/longmemeval-cleaned) provides, per sample, a set of
"haystack" conversation sessions (the long context) plus a question/answer.
We map each sample to one virtual user:
  - the sample's haystack turn contents -> that user's memory_stream (add)
  - the sample's haystack_sessions structure -> that user's turn_stream
    (user/assistant turns, for chat-replay's recall-before-answer workload)

Search queries are content-derived by the scenarios from the user's own
memory/turn content, so the sample's separate evaluation question is not
exposed as a search stream here.

When `n_users` exceeds the number of available samples, samples are replicated
onto additional virtual users (different user ids, same underlying content).
The virtual-user count we drive is what matters for load, not the dataset's
own sample count.

Dataset loading mirrors evaluation/retrieval_agent/longmemeval_test.py: prefer
the `datasets` loader, fall back to a direct JSON download via
huggingface_hub when the loader hits schema incompatibilities.

Data and attribution
--------------------
LongMemEval data is (c) its authors and licensed under the MIT license
(https://github.com/xiaowu0162/longmemeval). This adapter loads the data at
runtime from HuggingFace or from a user-supplied local path; the data is **not**
redistributed with LTM100. If you publish results obtained with this workload,
please cite the dataset:

    Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
    Interactive Memory," arXiv:2410.10813, 2024.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ltm100.common import MemoryItem, Turn, UserId
from ltm100.text import split_chunks as _split_chunks

logger = logging.getLogger(__name__)

_EVENT_TYPES = {
    "start_map": "dict",
    "string": "str",
    "number": "number",
    "boolean": "bool",
    "null": "NoneType",
}


def _collect_turn_contents(sample: dict[str, Any]) -> list[str]:
    """Flatten a sample's haystack_sessions into chunked turn contents."""
    return [content for _, contents in _iter_turns(sample) for content in contents]


def _iter_turns(sample: dict[str, Any]) -> Iterator[tuple[str, list[str]]]:
    """Yield each non-empty haystack turn's role and chunked contents."""
    for session in _iter_sessions(sample):
        yield from session


def _iter_sessions(
    sample: dict[str, Any],
) -> Iterator[list[tuple[str, list[str]]]]:
    """Yield non-empty haystack sessions without losing turn boundaries."""
    for session in sample.get("haystack_sessions", []) or []:
        turns: list[tuple[str, list[str]]] = []
        for turn in session or []:
            role = str(turn.get("role", "")).strip() or "user"
            content = str(turn.get("content", "")).strip()
            chunks = _split_chunks(content)
            if chunks:
                turns.append((role, chunks))
        if turns:
            yield turns


class LongMemEvalAdapter:
    """DatasetAdapter over the LongMemEval-cleaned HF dataset."""

    name = "longmemeval"

    def __init__(
        self,
        split: str = "longmemeval_s_cleaned",
        length: int | None = None,
        cache_dir: str | None = None,
        path: str | None = None,
    ) -> None:
        """If `path` is given, load directly from that local JSON file instead
        of downloading from HuggingFace. `split`/`cache_dir` are ignored when
        `path` is set."""
        self.split = split
        self.length = length
        self.cache_dir = cache_dir
        self.path = path
        self._records: list[dict[str, Any]] | None = None

    # -- loading -----------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records

        if self.path:
            self._records = self._load_local(self.path)
            return self._records

        self._records = self._load_hf()
        return self._records

    def _load_local(self, path: str) -> list[dict[str, Any]]:
        # Expand `~` so configs can use a home-relative path (e.g.
        # `path: ~/longmemeval/longmemeval_s_cleaned.json`).
        resolved = str(Path(path).expanduser())
        records = self._stream_local(resolved, path)
        if records is None:
            with open(resolved, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise TypeError(
                    f"Expected list data in {path}, got {type(raw).__name__}."
                )
            records = raw[: self.length] if self.length is not None else raw
        return self._normalize(records)

    def _stream_local(self, resolved: str, path: str) -> list[dict[str, Any]] | None:
        """Read at most `length` samples without materialising the document.

        json.load builds the whole file before any slice is taken: the 2.6 GB
        longmemeval_m split measured at 24 GB RSS to read twenty
        conversations, and every --procs shard pays that again. Returns None
        when ijson is absent, leaving the caller to fall back.
        """
        try:
            import ijson
        except ImportError:
            logger.warning(
                "ijson not installed; loading %s with json.load, which holds "
                "the whole file in memory. pip install ijson",
                resolved,
            )
            return None

        records: list[dict[str, Any]] = []
        with open(resolved, "rb") as f:
            # A non-array document yields no "item" events, which would look
            # like an empty dataset rather than the type error it is.
            first = next(ijson.parse(f), None)
            if first is None or first[1] != "start_array":
                raise TypeError(
                    f"Expected list data in {path}, got "
                    f"{_EVENT_TYPES.get(first[1], 'unknown') if first else 'empty'}."
                )
            f.seek(0)
            for item in ijson.items(f, "item", use_float=True):
                # Check before appending: length=0 must yield nothing, which is
                # what the json.load path's raw[:0] does.
                if self.length is not None and len(records) >= self.length:
                    break
                records.append(item)
        return records

    def _load_hf(self) -> list[dict[str, Any]]:
        split_file = (
            self.split if self.split.endswith(".json") else f"{self.split}.json"
        )
        records: list[dict[str, Any]] | None = None

        try:
            from datasets import load_dataset

            ds = load_dataset(
                "xiaowu0162/longmemeval-cleaned",
                split=self.split,
                cache_dir=self.cache_dir,
            )
            num_rows = len(ds)
            if self.length is not None:
                num_rows = min(self.length, num_rows)
            records = ds.select(range(num_rows)).to_list()
        # Loading can fail for backend-, cache-, authentication-, or schema-
        # specific reasons. Preserve the existing direct-download fallback.
        except Exception:  # noqa: BLE001
            from huggingface_hub import hf_hub_download

            data_path = hf_hub_download(
                repo_id="xiaowu0162/longmemeval-cleaned",
                repo_type="dataset",
                filename=split_file,
                cache_dir=self.cache_dir,
            )
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise TypeError(
                    f"Expected list data in {split_file}, got {type(raw).__name__}."
                )
            records = raw[: self.length] if self.length is not None else raw

        self._records = self._normalize(records)
        return self._records

    @staticmethod
    def _normalize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            rec = dict(record)
            rec["question"] = str(rec.get("question", ""))
            rec["answer"] = str(rec.get("answer", ""))
            rec.setdefault("question_type", "unknown")
            rec.setdefault("haystack_sessions", [])
            normalized.append(rec)
        return normalized

    # -- DatasetAdapter ----------------------------------------------------

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        """Map `n_users` virtual users onto samples, replicating if needed."""
        import random

        records = self._load()
        if not records:
            return []

        rng = random.Random(seed)
        # Stable assignment: user i -> sample (i % n_samples), with a shuffled
        # rotation so replicated users don't all map to the same order.
        indices = list(range(len(records)))
        rng.shuffle(indices)

        user_ids: list[UserId] = []
        for i in range(n_users):
            sample_idx = indices[i % len(indices)]
            user_ids.append(f"lme_user_{i:05d}_s{sample_idx:05d}")
        return user_ids

    def _sample_for_user(self, user: UserId) -> dict[str, Any]:
        """Resolve the dataset sample backing a virtual user id."""
        # user id format: lme_user_<uid>_s<sample_idx>
        try:
            sample_idx = int(user.rsplit("_s", 1)[1])
        except (IndexError, ValueError):
            sample_idx = 0
        records = self._load()
        return records[sample_idx % len(records)]

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        sample = self._sample_for_user(user)
        for role, chunks in _iter_turns(sample):
            for content in chunks:
                yield MemoryItem(content=content, producer=user, role=role)

    def turn_stream(self, user: UserId) -> Iterator[Turn]:
        """Yield haystack turns as (role, chunked items), in conversation order.

        Unlike `memory_stream` (which flattens all turn contents), this keeps
        the per-turn `role` and turn boundaries so a scenario can replay a
        chatbot-with-LTM workload: recall before a user turn, then ingest the
        user and following assistant turns. Long content is chunked the same
        way as in `memory_stream` (<=3000 chars on word boundaries)."""
        sample = self._sample_for_user(user)
        for role, chunks in _iter_turns(sample):
            items = [
                MemoryItem(content=content, producer=user, role=role)
                for content in chunks
            ]
            yield Turn(role=role, items=items)

    def session_stream(self, user: UserId) -> Iterator[list[Turn]]:
        """Yield each LongMemEval haystack session as one conversation."""
        sample = self._sample_for_user(user)
        for session in _iter_sessions(sample):
            yield [
                Turn(
                    role=role,
                    items=[
                        MemoryItem(content=content, producer=user, role=role)
                        for content in chunks
                    ],
                )
                for role, chunks in session
            ]


__all__ = ["LongMemEvalAdapter", "_collect_turn_contents", "_split_chunks"]
