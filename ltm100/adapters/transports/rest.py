"""REST transport over aiohttp for backends exposing an HTTP JSON API.

A thin JSON POST/GET client. Backends that ship only an SDK (no server)
implement LTMClient directly and ignore this layer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from ltm100.errors import annotate_error


class RestTransport:
    """Stateless JSON-over-HTTP transport with a shared session.

    Methods take an absolute path and return the parsed JSON body. The
    transport raises on HTTP error status, letting the caller decide how to
    surface failures (the runner records them as op errors).
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> None:
        # Strip trailing slash so paths join cleanly.
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retries = retries
        self.retry_backoff = retry_backoff
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> RestTransport:  # noqa: PYI034
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
            )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert self._session is not None, "transport not opened"
        attempt = 0
        while True:
            try:
                return await self._once(method, path, json=json, params=params)
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as error:
                # Only connection-level failures are retried. An HTTP error
                # status is a real answer from the server and belongs in the
                # results; retrying it would understate the error rate.
                if attempt >= self.retries:
                    self._annotate(error, method, path)
                    raise
                await asyncio.sleep(self.retry_backoff * (2**attempt))
                attempt += 1
            except Exception as error:
                self._annotate(error, method, path)
                raise

    async def request_text(self, method: str, path: str) -> str:
        """Like request(), but returns the body verbatim instead of parsing it.

        For endpoints that are not JSON -- a Prometheus exposition page is
        plain text. Same error semantics: HTTP >= 400 raises RestError.
        """
        assert self._session is not None, "transport not opened"
        attempt = 0
        while True:
            try:
                async with self._session.request(method, path) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RestError(
                            f"{method} {path} -> {resp.status}: {text[:500]}",
                            http_status=resp.status,
                        )
                    return text
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as error:
                # Connection-level retries only, for the same reason as request().
                if attempt >= self.retries:
                    self._annotate(error, method, path)
                    raise
                await asyncio.sleep(self.retry_backoff * (2**attempt))
                attempt += 1
            except Exception as error:
                self._annotate(error, method, path)
                raise

    def _annotate(self, error: Exception, method: str, path: str) -> None:
        category = None
        if isinstance(error, RestError):
            category = "http"
        elif isinstance(error, asyncio.TimeoutError):
            category = "timeout"
        elif isinstance(error, aiohttp.ClientConnectionError):
            category = "connection"
        elif isinstance(error, aiohttp.ClientPayloadError):
            category = "response"
        annotate_error(
            error,
            category=category,
            method=method,
            endpoint=path,
            http_status=error.http_status if isinstance(error, RestError) else None,
            timeout_seconds=self.timeout.total,
        )

    async def _once(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert self._session is not None, "transport not opened"
        async with self._session.request(
            method, path, json=json, params=params
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RestError(
                    f"{method} {path} -> {resp.status}: {text[:500]}",
                    http_status=resp.status,
                )
            if not text:
                return {}
            import json as _json

            try:
                return _json.loads(text)
            except _json.JSONDecodeError as error:
                annotate_error(error, category="response", http_status=resp.status)
                raise


class RestError(RuntimeError):
    """Raised when the server returns an HTTP error status."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


__all__ = ["RestError", "RestTransport"]
