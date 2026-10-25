"""Synthetic dataset adapter for fast, dependency-free load testing.

Generates deterministic per-user content from a seed, so a run can be driven
without any external dataset download. Useful for smoke runs, regression, and
comparing backends on identical synthetic load.

Each virtual user gets `memories_per_user` memory items. `content_chars` requests
filler characters, excluding the user/index prefix; it is not the total content
size, a token count, or an HTTP payload size. The legacy filler can undershoot
small requested lengths. Content is deterministic per effective corpus seed,
user, and index. An optional `corpus_seed` overrides the run seed for data only.
Search queries remain content-derived by the scenarios, with no separate query
stream exposed.

It is a real DatasetAdapter (registered as "synthetic"), not a test fixture.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator

from ltm100.common import MemoryItem, UserId


class SyntheticAdapter:
    """Deterministic synthetic dataset with no external dependencies."""

    name = "synthetic"

    def __init__(
        self,
        memories_per_user: int = 100,
        content_chars: int = 200,
        categories: int = 0,
        corpus_seed: int | None = None,
    ) -> None:
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
        self.content_chars = content_chars
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
        return {
            "generator": "legacy-filler-v1",
            "memories_per_user": self.memories_per_user,
            "content_chars": self.content_chars,
            "content_chars_scope": "filler",
            "categories": self.categories,
            "corpus_seed": self._effective_seed(seed),
            "corpus_seed_source": (
                "run_seed" if self.corpus_seed is None else "dataset"
            ),
        }

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
        for i in range(self.memories_per_user):
            content = f"{user} memory {i}: " + _filler(rng, self.content_chars)
            # cat_<i mod N>, the same naming the REST bench uses, so a filter
            # expression is comparable between the two harnesses.
            metadata = (
                {"category": f"cat_{i % self.categories}"} if self.categories else {}
            )
            yield MemoryItem(content=content, producer=user, metadata=metadata)


def _filler(rng: random.Random, n: int) -> str:
    words = [
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
    ]
    out = []
    while len(out) < n // 4 + 1:
        out.append(rng.choice(words))
    return " ".join(out)[:n]


__all__ = ["SyntheticAdapter"]
