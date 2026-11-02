"""Validated, immutable filler-length distributions for Synthetic corpora."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import accumulate

_OPTION = "synthetic.content_chars_distribution"


def _positive_chars(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{_OPTION}.{field} must be an integer >= 1")
    return value


@dataclass(frozen=True)
class UniformLengths:
    min_chars: int
    max_chars: int

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.min_chars, self.max_chars)

    def settings(self) -> dict:
        return {
            "type": "uniform",
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
        }


@dataclass(frozen=True)
class WeightedLengths:
    chars: tuple[int, ...]
    weights: tuple[float, ...]
    cumulative_weights: tuple[float, ...]

    def sample(self, rng: random.Random) -> int:
        return rng.choices(self.chars, cum_weights=self.cumulative_weights, k=1)[0]

    def settings(self) -> dict:
        return {
            "type": "weighted",
            "choices": [
                {"chars": chars, "weight": weight}
                for chars, weight in zip(self.chars, self.weights)
            ],
        }


def parse_distribution(value: object) -> UniformLengths | WeightedLengths:
    """Validate once, before generation, and copy caller-owned configuration."""
    if not isinstance(value, dict):
        raise ValueError(f"{_OPTION} must be a mapping or null")  # noqa: TRY004 - config error
    kind = value.get("type")
    if kind == "uniform":
        if set(value) != {"type", "min_chars", "max_chars"}:
            raise ValueError(
                f"{_OPTION} uniform requires only type/min_chars/max_chars"
            )
        minimum = _positive_chars(value["min_chars"], "min_chars")
        maximum = _positive_chars(value["max_chars"], "max_chars")
        if minimum > maximum:
            raise ValueError(f"{_OPTION}.min_chars must be <= max_chars")
        return UniformLengths(minimum, maximum)
    if kind == "weighted":
        if set(value) != {"type", "choices"}:
            raise ValueError(f"{_OPTION} weighted requires only type/choices")
        choices = value["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{_OPTION}.choices must be a non-empty list")
        chars = []
        seen = set()
        weights = []
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict) or set(choice) != {"chars", "weight"}:
                raise ValueError(
                    f"{_OPTION}.choices[{index}] requires only chars/weight"
                )
            size = _positive_chars(choice["chars"], f"choices[{index}].chars")
            if size in seen:
                raise ValueError(f"{_OPTION}.choices must use distinct chars values")
            weight = choice["weight"]
            if type(weight) not in (int, float):
                raise ValueError(
                    f"{_OPTION}.choices[{index}].weight must be a finite number > 0"
                )
            try:
                weight = float(weight)
            except OverflowError as error:
                raise ValueError(
                    f"{_OPTION}.choices[{index}].weight must be finite"
                ) from error
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"{_OPTION}.choices[{index}].weight must be a finite number > 0"
                )
            chars.append(size)
            seen.add(size)
            weights.append(weight)
        cumulative = tuple(accumulate(weights))
        if not math.isfinite(cumulative[-1]):
            raise ValueError(f"{_OPTION} total weight must be finite")
        # Keep sampling arithmetic out of the subnormal range: multiplying a
        # tiny total by random() can otherwise distort even equal weights.
        scale = max(weights)
        cumulative = tuple(accumulate(weight / scale for weight in weights))
        return WeightedLengths(tuple(chars), tuple(weights), cumulative)
    raise ValueError(f"{_OPTION}.type must be 'uniform' or 'weighted'")
