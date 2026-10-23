"""Bounded, backend-independent diagnostics for failed operations.

Only request context and generated messages enter reports. Exception strings,
response bodies, headers, query parameters, and tool arguments are excluded.
Enrichment is best-effort and never replaces an exception or its cause.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, dataclass
from http import HTTPStatus
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ErrorDetail:
    category: str
    message: str
    method: str | None = None
    endpoint: str | None = None
    http_status: int | None = None
    timeout_seconds: float | None = None
    tool_name: str | None = None
    tool_status: int | None = None

    def as_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


_MESSAGES = {
    "timeout": "Request timed out",
    "connection": "Connection failed",
    "response": "Invalid response",
    "mcp": "MCP tool call failed",
    "other": "Operation failed",
}


def error_detail(error: Exception) -> ErrorDetail:
    """Read transport diagnostics or classify a bounded explicit cause chain."""
    try:
        current: BaseException | None = error
        for _ in range(8):
            if current is None:
                break
            detail = getattr(current, "_ltm100_error_detail", None)
            if isinstance(detail, ErrorDetail):
                return detail
            if isinstance(current, (TimeoutError, asyncio.TimeoutError)):
                return ErrorDetail("timeout", _MESSAGES["timeout"])
            if isinstance(current, ConnectionError):
                return ErrorDetail("connection", _MESSAGES["connection"])
            if isinstance(current, json.JSONDecodeError):
                return ErrorDetail("response", _MESSAGES["response"])
            current = current.__cause__
    except Exception:  # noqa: BLE001 - diagnostics must preserve the original failure
        return ErrorDetail("other", _MESSAGES["other"])
    return ErrorDetail("other", _MESSAGES["other"])


def annotate_error(
    error: Exception,
    *,
    category: str | None = None,
    method: str | None = None,
    endpoint: str | None = None,
    http_status: int | None = None,
    timeout_seconds: float | None = None,
    tool_name: str | None = None,
    tool_status: int | None = None,
) -> None:
    """Attach serializable context without changing exception identity/type."""
    try:
        previous = error_detail(error)
        category = category or previous.category
        status = http_status if http_status is not None else previous.http_status
        message = _MESSAGES.get(category, _MESSAGES["other"])
        if category == "http":
            message = "HTTP request failed"
            if status is not None:
                try:
                    message = f"HTTP {status} {HTTPStatus(status).phrase}"
                except ValueError:
                    message = f"HTTP request failed (status {status})"
        timeout = timeout_seconds if category == "timeout" else None
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            timeout = None
        detail = ErrorDetail(
            category=category,
            message=message,
            method=method.upper()[:16] if method else previous.method,
            endpoint=urlsplit(endpoint).path[:256] if endpoint else previous.endpoint,
            http_status=status,
            timeout_seconds=timeout or previous.timeout_seconds,
            tool_name=tool_name[:80] if tool_name else previous.tool_name,
            tool_status=tool_status
            if tool_status is not None
            else previous.tool_status,
        )
        error._ltm100_error_detail = detail  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - diagnostics must preserve the original failure
        # A third-party exception may forbid attributes or expose a broken
        # property. Recording the original failure must still work.
        return
