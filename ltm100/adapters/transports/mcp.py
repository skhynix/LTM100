"""MCP transport over fastmcp for backends exposing memory tools as MCP tools.

A thin wrapper around `fastmcp.Client` that calls the backend's `add_memory`
and `search_memory` tools. Unlike the REST transport (one JSON request in,
one JSON body out), MCP tools are invoked by name with argument dicts, and
the per-user tenancy (org/project/user) is passed as tool arguments rather
than embedded in a request body or URL.

A single long-lived `fastmcp.Client` is shared across all calls; tenancy is
carried per-call in the tool arguments, so there is no per-user client churn.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import Client

from ltm100.errors import annotate_error, error_detail

logger = logging.getLogger(__name__)


class McpError(Exception):
    """Raised when an MCP tool call fails or returns an error response."""


class McpTransport:
    """MCP tool-call transport backed by a shared fastmcp.Client.

    The transport opens one client session for the whole run (lifespan matches
    the backend adapter's `__aenter__`/`__aexit__`). Tenancy is supplied by the
    caller on each `add`/`search` call via the tool arguments, not held here.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
    ) -> None:
        # Mount path of the MCP app (e.g. http://localhost:8080/mcp).
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Client | None = None

    async def __aenter__(self) -> McpTransport:  # noqa: PYI034
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        if self._client is None:
            # fastmcp.Client must be entered as an async context manager to
            # open its transport (initialize handshake) before any call_tool.
            client = Client(self.base_url, timeout=self.timeout)
            await client.__aenter__()
            self._client = client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Invoke an MCP tool by name and return its result data.

        Raises `McpError` if the client is closed or the tool returns an error
        `McpResponse` (status != 200). On success returns the tool's structured
        result (e.g. a `SearchResult` for `search_memory`)."""
        if self._client is None:
            error = McpError("MCP transport is not open")
            self._annotate(error, name)
            raise error
        try:
            result = await self._client.call_tool(name, arguments)
        except Exception as e:
            error = McpError(f"MCP call {name!r} failed: {e}")
            error.__cause__ = e
            self._annotate(error, name)
            raise error from e
        data = result.data
        # Tool failures come back as a McpResponse(status, message) rather
        # than raised. Distinguish it from a successful SearchResult, which
        # also has a `status` field but defaults to 0 (success) and carries a
        # `content` payload instead of a `message`. Only McpResponse has a
        # `message`; treat a non-200 status there as an error.
        if getattr(data, "message", None) is not None and not getattr(
            data, "content", None
        ):
            status = getattr(data, "status", 0)
            if status != 200:
                error = McpError(
                    f"MCP tool {name!r} returned status {status}: {data.message}"
                )
                self._annotate(error, name, tool_status=status)
                raise error
        return data

    def _annotate(
        self, error: McpError, name: str, *, tool_status: int | None = None
    ) -> None:
        category = error_detail(error).category
        annotate_error(
            error,
            category="mcp" if category == "other" else category,
            endpoint=self.base_url,
            timeout_seconds=self.timeout,
            tool_name=name,
            tool_status=tool_status,
        )


__all__ = ["McpError", "McpTransport"]
