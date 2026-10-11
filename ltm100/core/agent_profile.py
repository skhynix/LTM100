"""Request-policy and simulated timing settings for coding-agent replay."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _number(name: str, value: Any, *, upper: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"{name} must be a finite number in [0, {upper or 'inf'}]")


@dataclass(frozen=True)
class AgentProfile:
    """One uniform policy; transport and dataset configuration stay separate."""

    integration: str = "hook"
    hook_scope: str = "user-facing"
    search_probability: float = 0.25
    add_probability: float = 0.25
    agent_step_time: float = 0.0
    agent_step_time_variation: float | None = None
    tool_time: float = 0.0
    tool_time_variation: float | None = None
    tool_times: dict[str, float] = field(default_factory=dict)
    task_gap: float = 0.0
    task_gap_variation: float | None = None
    task_repeat: bool = True
    chunk_chars: int = 3000
    query_chars: int = 3000

    def __post_init__(self) -> None:
        if not isinstance(self.integration, str) or self.integration not in {
            "hook",
            "mcp",
            "hybrid",
        }:
            raise ValueError("integration must be hook, mcp, or hybrid")
        if not isinstance(self.hook_scope, str) or self.hook_scope not in {
            "all-steps",
            "user-facing",
        }:
            raise ValueError("hook_scope must be all-steps or user-facing")
        for name in ("search_probability", "add_probability"):
            _number(name, getattr(self, name), upper=1)
        if (
            self.integration == "mcp"
            and self.search_probability == 0
            and self.add_probability == 0
        ):
            raise ValueError("mcp requires at least one non-zero call probability")
        for name in ("agent_step_time", "tool_time", "task_gap"):
            _number(name, getattr(self, name))
            variation = getattr(self, f"{name}_variation")
            if variation is not None:
                _number(f"{name}_variation", variation, upper=1)
        if not isinstance(self.tool_times, dict):
            raise TypeError("tool_times must map tool names to durations")
        for name, duration in self.tool_times.items():
            if not isinstance(name, str) or not name or name == "finish":
                raise ValueError("tool_times requires non-empty ordinary tool names")
            _number(f"tool_times.{name}", duration)
        if not isinstance(self.task_repeat, bool):
            raise TypeError("task_repeat must be a boolean")
        for name in ("chunk_chars", "query_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def metadata(self) -> dict[str, Any]:
        return {"version": 1, **asdict(self)}

    @staticmethod
    def delay(rng: random.Random, mean: float, variation: float | None) -> float:
        """Use the same exponential/bounded-uniform convention as chat-replay."""
        if mean == 0:
            return 0.0
        if variation is None:
            return rng.expovariate(1.0 / mean)
        return rng.uniform(mean * (1 - variation), mean * (1 + variation))


def load_agent_profile(path: str | Path | None) -> AgentProfile:
    if path is None:
        return AgentProfile()
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise TypeError("agent profile must be a YAML mapping")
    version = raw.pop("version", 1)
    if isinstance(version, bool) or version != 1:
        raise ValueError("agent profile version must be 1")
    if any(not isinstance(key, str) for key in raw):
        raise TypeError("agent profile keys must be strings")
    unknown = set(raw) - {setting.name for setting in fields(AgentProfile)}
    if unknown:
        raise ValueError(f"unknown agent profile settings: {sorted(unknown)}")
    return AgentProfile(**raw)


__all__ = ["AgentProfile", "load_agent_profile"]
