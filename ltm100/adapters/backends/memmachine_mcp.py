"""MemMachine backend adapter over the MCP transport (hybrid lifecycle).

This adapter exposes MemMachine's `add_memory`/`search_memory` MCP tools under
the same `LTMClient` contract as the REST adapter, so a benchmark run can drive
the identical workload over MCP and compare it against REST.

Lifecycle is hybrid because MCP has no project create/delete tools:

  - `setup` / `teardown` (per-user tenant provisioning) go through the REST
    transport, exactly like `MemMachineClient`. Provisioning is out of
    measurement, so mixing transports here does not affect the measured
    add/search path.
  - `add` / `search` (the measured ops) go through the MCP transport's
    `add_memory` / `search_memory` tools.

Tenancy mapping matches the REST adapter: a virtual user maps to
`session_key = f"{org_prefix}/user_{UserId}"`, carried as MCP tool arguments
`org_id` / `proj_id` / `user_id`.

Two caveats versus the REST adapter, by design:
  - `add_memory` writes `types=ALL_MEMORY_TYPES` (episodic + semantic); the
    REST adapter is episodic-only. Semantic memory triggers LLM-based
    background processing, so MCP add latency is not directly comparable to
    REST add latency. This is inherent to the MCP tool and is documented, not
    worked around.
  - `add_memory` returns a success `McpResponse` with no ids, so `add`
    reports `n_items` as the number of items sent (not ids returned), unlike
    REST which counts returned uids.
"""

from __future__ import annotations

import logging
from typing import Any

from ltm100.adapters.backends.memmachine import _parse_episodes
from ltm100.adapters.transports.mcp import McpError, McpTransport
from ltm100.adapters.transports.rest import RestError, RestTransport
from ltm100.common import MemoryItem, QueryItem, ResultItem, UserId

logger = logging.getLogger(__name__)

# MCP add_memory writes all memory types (server-side ALL_MEMORY_TYPES). We
# cannot restrict it to episodic from the client, unlike the REST endpoint.
# Kept as a named constant for documentation; it is not sent to the tool.
_ALL_TYPES_SENT_SERVER_SIDE = "all"


class MemMachineMcpClient:
    """LTMClient adapter for MemMachine over MCP (hybrid REST lifecycle)."""

    name = "memmachine-mcp"

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        mcp_path: str = "/mcp",
        org_prefix: str = "ltm100",
        project_id: str = "",
        filter_by_producer: bool = False,
        timeout: float = 60.0,
        add_batch_size: int = 50,
    ) -> None:
        # Accepted only to refuse them by name: the tools take one proj_id per
        # call and no filter, so neither isolation-scope option can be honoured.
        if project_id or filter_by_producer:
            raise ValueError(
                "the MCP backend supports neither project_id nor "
                "filter_by_producer: search_memory takes no filter. Use the "
                "memmachine (REST) backend for isolation-scope arms."
            )
        self.org_prefix = org_prefix
        self.add_batch_size = add_batch_size
        # REST transport for setup/teardown (project create/delete).
        self._rest = RestTransport(base_url, timeout=timeout)
        # MCP transport for the measured add/search ops.
        mcp_url = base_url.rstrip("/") + "/" + mcp_path.strip("/")
        self._mcp = McpTransport(mcp_url, timeout=timeout)

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "MemMachineMcpClient":
        await self._rest.open()
        await self._mcp.open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._mcp.close()
        await self._rest.close()

    def _tenant(self, user: UserId) -> tuple[str, str]:
        """Map a virtual user to (org_id, project_id), same as the REST adapter."""
        return self.org_prefix, f"user_{user}"

    async def setup(self, users: list[UserId]) -> None:
        # Create one project per user via REST; 409 tolerated for reruns.
        for user in users:
            org_id, project_id = self._tenant(user)
            try:
                await self._rest.request(
                    "POST",
                    "/api/v2/projects",
                    json={
                        "org_id": org_id,
                        "project_id": project_id,
                        "description": f"ltm100 virtual user {user}",
                    },
                )
            except RestError as e:
                if "409" in str(e) or "already exists" in str(e).lower():
                    logger.debug("project %s/%s already exists", org_id, project_id)
                else:
                    raise

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        if not delete:
            return
        for user in users:
            org_id, project_id = self._tenant(user)
            try:
                await self._rest.request(
                    "POST",
                    "/api/v2/projects/delete",
                    json={"org_id": org_id, "project_id": project_id},
                )
            except RestError as e:
                logger.debug("delete project %s/%s failed: %s", org_id, project_id, e)

    # -- LTMClient (measured ops over MCP) --------------------------------

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        org_id, project_id = self._tenant(user)
        # add_memory takes a single content string per call, so we issue one
        # tool call per memory item (the MCP tool has no batch form). The
        # tool returns a success McpResponse with no ids.
        # The add_memory tool takes no metadata field, so carrying any here
        # would drop it silently and then a metadata filter would select
        # nothing for reasons invisible in the results.
        if any(item.metadata for item in items):
            raise ValueError(
                "the MCP backend cannot store item metadata: add_memory has no "
                "field for it. Use the memmachine (REST) backend for a corpus "
                "that sets metadata, e.g. the synthetic dataset's categories."
            )
        uids: list[str] = []
        for item in items:
            payload = {
                "content": item.content,
                "org_id": org_id,
                "proj_id": project_id,
                "user_id": user,
            }
            if item.producer is not None:
                payload["user_id"] = item.producer
            try:
                await self._mcp.call_tool("add_memory", payload)
            except McpError:
                raise
            # No uid is returned; record a placeholder so n_items counts sent
            # items consistently with the caller's expectation of one id per
            # item. The runner records n_items = len(uids) = len(items).
            uids.append("")
        return uids

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        org_id, project_id = self._tenant(user)
        # search_memory exposes neither knob. Ignoring them would report a
        # baseline search under the label of a filtered or expanded one.
        if query.expand_context or query.filter:
            raise ValueError(
                "the MCP backend supports neither --expand nor --filter: "
                "search_memory takes only query and top_k. Use the memmachine "
                "(REST) backend for those arms."
            )
        payload = {
            "query": query.query,
            "top_k": query.top_k,
            "org_id": org_id,
            "proj_id": project_id,
            "user_id": user,
        }
        data = await self._mcp.call_tool("search_memory", payload)
        return _parse_episodes(_to_search_dict(data))


def _to_search_dict(data: Any) -> dict[str, Any]:
    """Normalize a fastmcp search_memory result into the dict shape that
    `_parse_episodes` expects (the same `content.episodic_memory...` path the
    REST `/memories/search` response uses).

    fastmcp returns a pydantic `SearchResult` (or a dict-like Root); we reach
    its attributes defensively so a plain dict also works."""
    # If it is already a dict, use it directly.
    if isinstance(data, dict):
        return data
    content = getattr(data, "content", None)
    if content is None:
        return {}
    # content is a SearchResultContent pydantic model; serialize its
    # episodic_memory branch into the dict shape _parse_episodes reads.
    em = getattr(content, "episodic_memory", None)
    if em is None:
        return {}
    ltm = getattr(em, "long_term_memory", None)
    if ltm is None:
        return {"content": {"episodic_memory": {}}}
    episodes = getattr(ltm, "episodes", []) or []
    return {
        "content": {
            "episodic_memory": {
                "long_term_memory": {
                    "episodes": [_episode_to_dict(ep) for ep in episodes],
                }
            }
        }
    }


def _episode_to_dict(ep: Any) -> dict[str, Any]:
    if isinstance(ep, dict):
        return ep
    return {
        "content": getattr(ep, "content", ""),
        "score": getattr(ep, "score", None),
        "uid": getattr(ep, "uid", None) or getattr(ep, "id", None),
        "metadata": getattr(ep, "metadata", {}) or {},
    }


__all__ = ["MemMachineMcpClient"]
