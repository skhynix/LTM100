"""Synthetic dataset adapter for fast, dependency-free load testing.

Generates deterministic per-user content from a seed, so a run can be driven
without any external dataset download. Useful for smoke runs, regression, and
comparing backends on identical synthetic load.

Each virtual user gets `memories_per_user` memory items. `content_chars` requests
filler characters, excluding the user/index prefix; it is not the total content
size, a token count, or an HTTP payload size. The legacy filler can undershoot
small requested lengths. Optional `content_chars_distribution` instead samples
an exact filler length per corpus item, using a separate length RNG. Content
is deterministic per effective corpus seed,
user, and index. An optional `corpus_seed` overrides the run seed for data only.
Search queries remain content-derived by the scenarios, with no separate query
stream exposed.

It is a real DatasetAdapter (registered as "synthetic"), not a test fixture.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from enum import Enum

from ltm100.adapters.datasets.synthetic_lengths import parse_distribution
from ltm100.common import MemoryItem, UserId


class _DefaultContentChars(Enum):
    # Distinguish omission from an explicitly supplied value, including 200.
    VALUE = 200


_WORDS = (
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "a",
    "lazy",
    "dog",
    "while",
    "memory",
    "systems",
    "store",
    "and",
    "retrieve",
)


class SyntheticAdapter:
    """Deterministic synthetic dataset with no external dependencies."""

    name = "synthetic"

    def __init__(
        self,
        memories_per_user: int = 100,
        content_chars: int | _DefaultContentChars = _DefaultContentChars.VALUE,
        categories: int = 0,
        corpus_seed: int | None = None,
        content_chars_distribution: dict | None = None,
    ) -> None:
        if (
            content_chars_distribution is not None
            and content_chars is not _DefaultContentChars.VALUE
        ):
            raise ValueError(
                "synthetic.content_chars and content_chars_distribution are mutually exclusive"
            )
        if content_chars is _DefaultContentChars.VALUE:
            content_chars = 200
        for name, value, minimum in (
            ("memories_per_user", memories_per_user, 1),
            ("content_chars", content_chars, 1),
            ("categories", categories, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"synthetic.{name} must be an integer >= {minimum}")
        if corpus_seed is not None and type(corpus_seed) is not int:
            raise ValueError("synthetic.corpus_seed must be an integer or null")
        self.memories_per_user = memories_per_user
        self._length_distribution = (
            parse_distribution(content_chars_distribution)
            if content_chars_distribution is not None
            else None
        )
        self.content_chars = (
            content_chars if self._length_distribution is None else None
        )
        self.corpus_seed = corpus_seed
        # The runner initializes the run seed through users() before consuming
        # any corpus or query stream. Without an override, direct use before
        # users() retains seed 0.
        self._seed = self._effective_seed(0)
        # A field for --filter to select on. 0 leaves metadata empty, so the
        # corpus is byte-identical to one built without this option; N spreads
        # items over N values, so a single-value filter selects about 1/N.
        self.categories = categories

    def _effective_seed(self, seed: int) -> int:
        return seed if self.corpus_seed is None else self.corpus_seed

    def generation_settings(self, *, seed: int) -> dict:
        """Describe resolved generation settings without consuming the corpus."""
        settings = {
            "generator": (
                "legacy-filler-v1"
                if self._length_distribution is None
                else "distributed-filler-v1"
            ),
            "memories_per_user": self.memories_per_user,
            "content_chars": self.content_chars,
            "content_chars_scope": "filler",
            "categories": self.categories,
            "corpus_seed": self._effective_seed(seed),
            "corpus_seed_source": (
                "run_seed" if self.corpus_seed is None else "dataset"
            ),
        }
        if self._length_distribution is not None:
            settings["content_chars_distribution"] = (
                self._length_distribution.settings()
            )
        return settings

    def _user_rng(self, user: UserId, seed: int) -> random.Random:
        if seed != 0:
            # Include the full signed seed, independent of Python hash salts.
            digest = hashlib.sha256(f"{seed}:{user}".encode()).digest()
            return random.Random(int.from_bytes(digest, "big"))
        # Preserve the historical seed-zero corpus byte for byte.
        h = 0
        for ch in user.encode("utf-8"):
            h = (h ^ ch) * 1_000_003 & 0xFFFFFFFF
        return random.Random(h)

    def users(self, n_users: int, *, seed: int = 0) -> list[UserId]:
        self._seed = self._effective_seed(seed)
        return [f"syn_user_{i:05d}" for i in range(n_users)]

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        rng = self._user_rng(user + "::mem", self._seed)
        distribution = self._length_distribution
        length_rng = (
            self._user_rng(user + "::length", self._seed)
            if distribution is not None
            else None
        )
        for i in range(self.memories_per_user):
            if distribution is None:
                filler = _filler(rng, self.content_chars)
            else:
                filler = _exact_filler(rng, distribution.sample(length_rng))
            content = f"{user} memory {i}: " + filler
            # cat_<i mod N>, the same naming the REST bench uses, so a filter
            # expression is comparable between the two harnesses.
            metadata = (
                {"category": f"cat_{i % self.categories}"} if self.categories else {}
            )
            yield MemoryItem(content=content, producer=user, metadata=metadata)


def _filler(rng: random.Random, n: int) -> str:
    out = []
    while len(out) < n // 4 + 1:
        out.append(rng.choice(_WORDS))
    return " ".join(out)[:n]


def _exact_filler(rng: random.Random, n: int) -> str:
    """Fill the sampled size in linear time; the legacy path stays unchanged."""
    out = []
    length = 0
    while length < n:
        word = rng.choice(_WORDS)
        length += len(word) + bool(out)
        out.append(word)
    return " ".join(out)[:n]


__all__ = ["SyntheticAdapter"]
