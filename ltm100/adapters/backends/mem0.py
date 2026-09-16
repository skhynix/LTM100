"""Mem0 OSS REST backend adapter.

The self-hosted Mem0 server exposes unversioned endpoints.  Each LTM100
virtual user maps to a Mem0 ``user_id`` so all operations remain tenant scoped.
"""

from __future__ import annotations

from typing import Any

from ltm100.adapters.transports.rest import RestTransport
from ltm100.common import MemoryItem, QueryItem, ResultItem, UserId


class Mem0Client:
    """LTMClient adapter for the self-hosted Mem0 OSS REST server."""

    name = "mem0"

    def __init__(
        self,
        base_url: str = "http://localhost:8888",
        *,
        user_prefix: str = "ltm100",
        api_key: str | None = None,
        timeout: float = 60.0,
        retries: int = 0,
        infer: bool = False,
    ) -> None:
        self.user_prefix = user_prefix
        self.infer = infer
        headers = {"X-API-Key": api_key} if api_key else None
        self._transport = RestTransport(
            base_url, headers=headers, timeout=timeout, retries=retries
        )

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> Mem0Client:  # noqa: PYI034
        await self._transport.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._transport.close()

    def _user_id(self, user: UserId) -> str:
        return f"{self.user_prefix}_user_{user}"

    # -- LTMClient --------------------------------------------------------

    async def setup(self, users: list[UserId]) -> None:
        # Mem0 creates user scopes lazily on the first add.
        return None

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        user_id = self._user_id(user)
        uids: list[str] = []
        # Mem0 accepts one metadata object per request. Send one item at a time
        # so per-item metadata and the one-input/one-id contract are preserved.
        for item in items:
            message: dict[str, Any] = {
                "role": item.role or "user",
                "content": item.content,
            }
            if item.producer:
                message["name"] = item.producer
            metadata = dict(item.metadata)
            if item.timestamp:
                metadata["timestamp"] = item.timestamp
            payload: dict[str, Any] = {
                "messages": [message],
                "user_id": user_id,
                "infer": self.infer,
            }
            if metadata:
                payload["metadata"] = metadata
            response = await self._transport.request(
                "POST", "/memories", json=payload
            )
            for result in response.get("results", []):
                uids.append(str(result.get("id", "")))
        return uids

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        if query.expand_context:
            raise ValueError("Mem0 does not support expand_context")
        filters: dict[str, Any] = {"user_id": self._user_id(user)}
        if query.filter:
            key, value = _parse_filter(query.filter)
            filters[key] = value
        response = await self._transport.request(
            "POST",
            "/search",
            json={"query": query.query, "filters": filters, "top_k": query.top_k},
        )
        rows = response.get("results", []) if isinstance(response, dict) else []
        return [
            ResultItem(
                content=row.get("memory") or row.get("data") or "",
                score=row.get("score"),
                uid=str(row.get("id")) if row.get("id") is not None else None,
                metadata=row.get("metadata", {}) or {},
            )
            for row in rows
        ]

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        if not delete:
            return
        for user in users:
            await self._transport.request(
                "DELETE", "/memories", params={"user_id": self._user_id(user)}
            )


# -- helpers -----------------------------------------------------------------


def _parse_filter(value: str) -> tuple[str, str]:
    """Translate LTM100's ``metadata.key=value`` exact-match syntax to Mem0."""
    if "=" not in value:
        raise ValueError("Mem0 filter must use key=value syntax")
    key, expected = value.split("=", 1)
    key = key.strip()
    expected = expected.strip()
    key = key.removeprefix("metadata.")
    if not key or not expected:
        raise ValueError("Mem0 filter must have a non-empty key and value")
    return key, expected


__all__ = ["Mem0Client"]
