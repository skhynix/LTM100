"""Operation and scenario types for the load core.

A Scenario decides, per virtual user, the *op mix* and *emit schedule* (op
order, think time / inter-arrival). Both closed and open load models share
the same `Op` primitive; the difference is how the runner consumes the plan.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ltm100.common import DatasetAdapter, MemoryItem, QueryItem, UserId


class OpType(str, enum.Enum):
    ADD = "add"
    SEARCH = "search"


@dataclass(frozen=True)
class Op:
    """A single scheduled request for a user."""

    type: OpType
    # Populated for ADD:
    items: list[MemoryItem] = field(default_factory=list)
    # Populated for SEARCH:
    query: QueryItem | None = None
    # Wall-clock pause (seconds) BEFORE emitting this op. Closed model: think
    # time between ops. Open model: inter-arrival gap. 0 = back-to-back.
    delay: float = 0.0
    # Optional workload-profile group. Empty for ungrouped scenarios.
    group: str = ""
    # Concurrent chat lane within one backend user. None for other workloads.
    session_id: int | None = None
    # Optional provenance; avoid allocating a mapping for ordinary workloads.
    context: dict[str, Any] | None = None


@dataclass(frozen=True)
class Wait:
    """A timeline pause with no backend request or operation-budget cost."""

    delay: float = 0.0


@dataclass(frozen=True)
class OpResult:
    """Result of executing an Op, recorded by the runner."""

    type: OpType
    user_id: UserId
    started_at: float  # epoch seconds
    ended_at: float  # epoch seconds
    status: str  # "ok" | "error" | "rejected"
    error_kind: str = ""
    n_items: int = 0  # items stored (add) or results returned (search)
    group: str = ""  # workload-profile group, when configured
    session_id: int | None = None
    context: dict[str, Any] | None = None


class Scenario(Protocol):
    """Defines how a virtual user emits requests."""

    name: str

    def plan(
        self,
        user: UserId,
        dataset: DatasetAdapter,
        rng_state: dict[str, Any],
    ) -> Iterator[Op | Wait]:
        """Yield the sequence of ops for this user, in order.

        `rng_state` carries the seeded RNG state so schedules are
        reproducible under a fixed seed. Implementations should draw all
        randomness from this state.
        """
        ...

    def validate(self, dataset: DatasetAdapter) -> None:
        """Optional pre-run check that the dataset supports this scenario.

        Implementations that need a specific dataset capability (e.g. a
        `turn_stream`) override this to raise `ValueError` early, so a
        misconfigured run fails loudly before any user runs. Default: no-op.
        """
        ...


__all__ = ["Op", "OpResult", "OpType", "Scenario", "Wait"]
