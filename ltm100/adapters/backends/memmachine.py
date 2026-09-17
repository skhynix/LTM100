"""MemMachine backend adapter (REST /api/v2).

MemMachine's multi-tenancy is `session_key = f"{org_id}/{project_id}"`; every
add/search is scoped by it. We map each LTM100 virtual user to one project:

    UserId  ->  org_id, project_id

All requests carry that pair so a user only ever sees their own memories. The
adapter is async (aiohttp via RestTransport) and lives entirely behind the
LTMClient Protocol, so the load core never touches MemMachine specifics.

Endpoints used:
  POST /api/v2/projects           create a project (per-user tenant)
  POST /api/v2/projects/get       look up a shared project before creating it
  POST /api/v2/projects/delete    delete a project
  POST /api/v2/memories           add memories (episodic)
  POST /api/v2/memories/search    search memories (episodic)
  GET  /api/v2/health             readiness check
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ltm100.adapters.transports.rest import RestError, RestTransport
from ltm100.common import MemoryItem, QueryItem, ResultItem, UserId

logger = logging.getLogger(__name__)

# We only ingest/search episodic memory in this benchmark.
_PRODUCER_FIELD = "producer_id"
_EPISODIC_TYPES = ["episodic"]


class MemMachineClient:
    """LTMClient adapter for MemMachine over REST."""

    name = "memmachine"

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        org_prefix: str = "ltm100",
        project_id: str = "",
        filter_by_producer: bool = False,
        timeout: float = 60.0,
        add_batch_size: int = 50,
        retries: int = 0,
    ) -> None:
        self.org_prefix = org_prefix
        self.project_id = project_id
        self.filter_by_producer = filter_by_producer
        self.add_batch_size = add_batch_size
        self._transport = RestTransport(base_url, timeout=timeout, retries=retries)

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "MemMachineClient":
        await self._transport.open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._transport.close()

    def _tenant(self, user: UserId) -> tuple[str, str]:
        """Map a virtual user to (org_id, project_id).

        By default every virtual user is pinned under a single org with the
        user id as the project id, so a run's users share an org but have
        distinct projects, giving per-user memory isolation.

        Set `project_id` and they all share that one project instead. That is
        a different measurement, not a variant of the same one: MemMachine
        partitions its vector collection by a key derived from
        `org_id/project_id` and builds it with `m=0, payload_m=16` -- no global
        HNSW links, only the per-value links Qdrant adds for each indexed field
        (the partition key among them) to each segment's one graph. A partition
        under the full-scan threshold is searched exactly instead. So many
        per-user projects and one shared project take different search paths."""
        if self.project_id:
            return self.org_prefix, self.project_id
        return self.org_prefix, f"user_{user}"

    async def health(self) -> dict[str, Any]:
        return await self._transport.request("GET", "/api/v2/health")

    # -- LTMClient --------------------------------------------------------

    async def setup(self, users: list[UserId]) -> None:
        # The filter grammar has no escape inside a quoted string (a literal is
        # '[^']*'), so an id carrying a quote would end the producer scope
        # early. No dataset produces one; refuse rather than escape.
        if self.filter_by_producer:
            quoted = [u for u in users if "'" in u]
            if quoted:
                raise ValueError(
                    f"filter_by_producer cannot scope user ids containing a quote: {quoted[:3]}"
                )
        # Create one project per user; 409 (already exists) is acceptable so
        # reruns against the same tenant don't fail. When every user shares one
        # project, one create covers them all -- doing it per user would send
        # N-1 redundant creates, and those land on the server's own request
        # counters.
        for user in users[:1] if self.project_id else users:
            org_id, project_id = self._tenant(user)
            # Re-creating an identical project answers 201, so a successful
            # create does not show this run made it. Ask first: a shared
            # project found here predates the run and is left on exit.
            if self.project_id and await self._project_exists(org_id, project_id):
                logger.debug("shared project %s/%s already exists", org_id, project_id)
                continue
            await self._create_project(org_id, project_id, user)

    async def _project_exists(self, org_id: str, project_id: str) -> bool:
        try:
            await self._transport.request(
                "POST",
                "/api/v2/projects/get",
                json={"org_id": org_id, "project_id": project_id},
            )
            return True
        except RestError as e:
            if "-> 404" in str(e):
                return False
            raise

    # Seconds before retrying a failed create of a shared project.
    _create_retry_delay = 0.5
    # Whether setup() created the shared project rather than finding it; a
    # shared project that predates the run is not teardown()'s to delete.
    _created_shared = False

    async def _create_project(self, org_id: str, project_id: str, user: UserId) -> None:
        body = {
            "org_id": org_id,
            "project_id": project_id,
            "description": f"ltm100 virtual user {user}",
        }
        for attempt in range(2):
            try:
                await self._transport.request("POST", "/api/v2/projects", json=body)
                self._created_shared = bool(self.project_id)
                return
            except RestError as e:
                if "409" in str(e) or "already exists" in str(e).lower():
                    logger.debug("project %s/%s already exists", org_id, project_id)
                    return
                # With --procs N every shard creates the shared project at
                # once, and the server can answer a racing create with a 500.
                # Retrying succeeds, since re-creating an identical project
                # answers 201. Per-user projects never race, so a 500 there is
                # a real failure.
                if not self.project_id or attempt:
                    raise
                logger.debug("create %s/%s failed, retrying once: %s", org_id, project_id, e)
                await asyncio.sleep(self._create_retry_delay)

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        org_id, project_id = self._tenant(user)
        uids: list[str] = []
        for start in range(0, len(items), self.add_batch_size):
            batch = items[start : start + self.add_batch_size]
            payload = {
                "org_id": org_id,
                "project_id": project_id,
                "types": _EPISODIC_TYPES,
                "messages": [_to_message(it) for it in batch],
            }
            resp = await self._transport.request("POST", "/api/v2/memories", json=payload)
            for r in resp.get("results", []):
                uids.append(r.get("uid", ""))
        return uids

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        org_id, project_id = self._tenant(user)
        payload: dict[str, Any] = {
            "org_id": org_id,
            "project_id": project_id,
            "query": query.query,
            "top_k": query.top_k,
            "types": _EPISODIC_TYPES,
        }
        # Omitted rather than sent as 0/"" so a default run's payload is
        # unchanged and the server applies its own defaults.
        if query.expand_context:
            payload["expand_context"] = query.expand_context
        # With users sharing a project, `producer_id` is what separates them --
        # AND-ed into the caller's filter rather than replacing it, so a
        # tenancy arm can still carry a metadata filter and the two stay
        # separable. `producer_id` is a server-side field with a vector-store
        # index behind it; user metadata under `m.` has none, so the two are
        # not comparable as filters.
        #
        # The caller's filter is parenthesised because AND binds tighter than
        # OR server-side: `producer_id = 'u' AND a OR b` parses as
        # `(producer_id = 'u' AND a) OR b`, and anything matching b comes back
        # whoever produced it.
        search_filter = query.filter
        if self.filter_by_producer:
            scope = f"{_PRODUCER_FIELD} = '{user}'"
            search_filter = f"{scope} AND ({search_filter})" if search_filter else scope
        if search_filter:
            payload["filter"] = search_filter
        resp = await self._transport.request("POST", "/api/v2/memories/search", json=payload)
        return _parse_episodes(resp)

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        if not delete:
            return
        if self.project_id and not self._created_shared:
            logger.info("leaving %s/%s: it existed before this run",
                        self.org_prefix, self.project_id)
            return
        # One delete when the project is shared, for the same reason as setup.
        for user in users[:1] if self.project_id else users:
            org_id, project_id = self._tenant(user)
            try:
                await self._transport.request(
                    "POST",
                    "/api/v2/projects/delete",
                    json={"org_id": org_id, "project_id": project_id},
                )
            except RestError as e:
                logger.debug("delete project %s/%s failed: %s", org_id, project_id, e)


# -- helpers -----------------------------------------------------------------


def _to_message(item: MemoryItem) -> dict[str, Any]:
    msg: dict[str, Any] = {"content": item.content}
    if item.producer is not None:
        msg["producer"] = item.producer
    if item.role:
        msg["role"] = item.role
    if item.timestamp:
        msg["timestamp"] = item.timestamp
    if item.metadata:
        # MemMachine metadata values must be strings.
        msg["metadata"] = {k: str(v) for k, v in item.metadata.items()}
    return msg


def _parse_episodes(resp: dict[str, Any]) -> list[ResultItem]:
    """Extract long-term episodic episodes from a /memories/search response."""
    content = resp.get("content", {}) or {}
    em = content.get("episodic_memory")
    if not em:
        return []
    ltm = em.get("long_term_memory", {}) or {}
    episodes = ltm.get("episodes", []) or []
    results: list[ResultItem] = []
    for ep in episodes:
        results.append(
            ResultItem(
                content=ep.get("content", ""),
                score=ep.get("score"),
                uid=ep.get("uid") or ep.get("id"),
                metadata=ep.get("metadata", {}) or {},
            )
        )
    return results


__all__ = ["MemMachineClient"]
