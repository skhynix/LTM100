"""Tests for the Mem0 OSS REST adapter against a small fake server."""

from __future__ import annotations

import pytest
from aiohttp import web

from ltm100.adapters.backends.mem0 import Mem0Client
from ltm100.common import MemoryItem, QueryItem

REQUESTS = web.AppKey("requests", list)


def _make_app() -> web.Application:
    store: dict[str, list[dict]] = {}
    app = web.Application()
    app[REQUESTS] = []

    async def add(request: web.Request) -> web.Response:
        body = await request.json()
        app[REQUESTS].append(("add", body, dict(request.headers)))
        user_id = body["user_id"]
        results = []
        for message in body["messages"]:
            uid = f"{user_id}-{len(store.setdefault(user_id, []))}"
            row = {
                "id": uid,
                "memory": message["content"],
                "metadata": body.get("metadata", {}),
                "score": 0.9,
            }
            store[user_id].append(row)
            results.append({**row, "event": "ADD"})
        return web.json_response({"results": results})

    async def search(request: web.Request) -> web.Response:
        body = await request.json()
        app[REQUESTS].append(("search", body, dict(request.headers)))
        filters = body["filters"]
        rows = store.get(filters["user_id"], [])
        for key, value in filters.items():
            if key != "user_id":
                rows = [r for r in rows if r["metadata"].get(key) == value]
        return web.json_response({"results": rows[: body["top_k"]]})

    async def delete(request: web.Request) -> web.Response:
        user_id = request.query["user_id"]
        app[REQUESTS].append(("delete", user_id, dict(request.headers)))
        store.pop(user_id, None)
        return web.json_response({"message": "deleted"})

    app.router.add_post("/memories", add)
    app.router.add_post("/search", search)
    app.router.add_delete("/memories", delete)
    return app


@pytest.fixture
async def mem0_server():
    app = _make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", app
    await runner.cleanup()


@pytest.mark.asyncio
async def test_setup_is_lazy_and_user_mapping_is_stable(mem0_server):
    url, app = mem0_server
    async with Mem0Client(url) as client:
        await client.setup(["u0", "u1"])
        assert client._user_id("u0") == "ltm100_user_u0"
        assert client._user_id("u1") == "ltm100_user_u1"
    assert app[REQUESTS] == []


@pytest.mark.asyncio
async def test_add_uses_raw_storage_and_preserves_item_metadata(mem0_server):
    url, app = mem0_server
    async with Mem0Client(url, api_key="secret") as client:
        ids = await client.add(
            "u0",
            [MemoryItem("hello", role="assistant", metadata={"category": "a"})],
        )
    assert ids == ["ltm100_user_u0-0"]
    _, body, headers = app[REQUESTS][0]
    assert body["infer"] is False
    assert body["messages"] == [{"role": "assistant", "content": "hello"}]
    assert body["metadata"] == {"category": "a"}
    assert headers["X-API-Key"] == "secret"


@pytest.mark.asyncio
async def test_search_is_user_scoped_and_respects_top_k(mem0_server):
    url, _ = mem0_server
    async with Mem0Client(url) as client:
        await client.add("u0", [MemoryItem(f"zero-{i}") for i in range(3)])
        await client.add("u1", [MemoryItem("one-only")])
        results = await client.search("u0", QueryItem("zero", top_k=2))
    assert [r.content for r in results] == ["zero-0", "zero-1"]
    assert all(r.uid and r.score == 0.9 for r in results)


@pytest.mark.asyncio
async def test_metadata_filter_is_translated(mem0_server):
    url, app = mem0_server
    async with Mem0Client(url) as client:
        await client.add("u0", [MemoryItem("a", metadata={"category": "cat_1"})])
        await client.add("u0", [MemoryItem("b", metadata={"category": "cat_2"})])
        results = await client.search(
            "u0", QueryItem("q", filter="metadata.category=cat_2")
        )
    assert [r.content for r in results] == ["b"]
    search_body = next(body for kind, body, _ in app[REQUESTS] if kind == "search")
    assert search_body["filters"] == {
        "user_id": "ltm100_user_u0",
        "category": "cat_2",
    }


@pytest.mark.asyncio
async def test_expand_context_is_rejected(mem0_server):
    url, _ = mem0_server
    async with Mem0Client(url) as client:
        with pytest.raises(ValueError, match="expand_context"):
            await client.search("u0", QueryItem("q", expand_context=1))


@pytest.mark.asyncio
async def test_teardown_deletes_only_requested_user(mem0_server):
    url, _ = mem0_server
    async with Mem0Client(url) as client:
        await client.add("u0", [MemoryItem("zero")])
        await client.add("u1", [MemoryItem("one")])
        await client.teardown(["u0"], delete=True)
        assert await client.search("u0", QueryItem("q")) == []
        assert [r.content for r in await client.search("u1", QueryItem("q"))] == [
            "one"
        ]
