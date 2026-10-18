"""Memory-type selection is an adapter setting, not a workload policy."""

from __future__ import annotations

from typing import Any

DEFAULT_TYPES = ("episodic",)
OPTIONS = ("add_memory_types", "search_memory_types")


def normalize_types(value: object, option: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            not isinstance(item, str) or item not in DEFAULT_TYPES + ("semantic",)
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{option} must be a non-empty list of unique memory types: episodic, semantic"
        )
    return tuple(value)


def validate_isolation(types: tuple[str, ...], filter_by_producer: bool) -> None:
    if "semantic" in types and filter_by_producer:
        raise ValueError(
            "semantic memory does not support filter_by_producer in this adapter; "
            "use episodic-only selection or project-per-user isolation"
        )


def memory_type_settings(name: str, options: dict[str, Any]) -> dict | None:
    """Validate config and describe request selection, not server completion."""
    if name != "memmachine":
        if any(key in options for key in OPTIONS):
            raise ValueError(
                f"{name} does not support add_memory_types/search_memory_types; "
                "omit these options to preserve its existing behavior"
            )
        return None
    resolved = {
        operation: list(normalize_types(options.get(key, DEFAULT_TYPES), key))
        for operation, key in zip(("add", "search"), OPTIONS)
    }
    validate_isolation(
        tuple(resolved["add"] + resolved["search"]),
        options.get("filter_by_producer", False),
    )
    return {
        "requested": {
            operation: list(options[key]) if key in options else None
            for operation, key in zip(("add", "search"), OPTIONS)
        },
        "resolved_request_types": resolved,
    }
