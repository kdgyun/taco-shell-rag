import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from taco_shell_rag.core.contracts import Document, Snapshot
from tests.mcp_fixture import BearerAuth, create_http_app, create_server


async def test_actual_mcp_session_search_and_document(kb):
    class Notes:
        id = "notes"

        def scan(self, checkpoint):
            return Snapshot([Document("policy", self.id, "refund policy")], "first")

    kb.sources = {"notes": Notes()}
    kb.sync()
    async with Client(create_server(kb)) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {"search", "get_document", "status"}
        result = await client.call_tool("search", {"query": "refund"})
        assert not result.is_error
        assert "refund policy" in str(result.content)
        data = result.structured_content
        chunk = data["results"][0]["chunk"]
        document = await client.call_tool(
            "get_document",
            {
                "document_key": chunk["document_key"],
                "version": chunk["version"],
            },
        )
        assert not document.is_error
        assert "refund policy" in str(document.content)
        failed = await client.call_tool("search", {"query": "refund", "source_id": "foreign"})
        assert failed.is_error


async def test_http_bearer_boundary():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    token = "a" * 64
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=BearerAuth(app, token)), base_url="http://test"
    ) as client:
        assert (await client.post("/mcp")).status_code == 401
        assert (
            await client.post("/mcp", headers={"Authorization": f"Bearer {token}"})
        ).status_code == 200
    with pytest.raises(ValueError, match="32"):
        BearerAuth(app, "short")


@pytest.mark.parametrize("url", ["http://localhost/mcp", "http://127.0.0.1:8080/mcp"])
async def test_authenticated_http_mcp_handshake_and_tools(kb, url):
    token = "a" * 64
    app = create_http_app(kb, token)
    async with app.app.router.lifespan_context(app.app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            headers={"Authorization": "Bearer " + token},
        ) as http:
            transport = streamable_http_client(url, http_client=http)
            async with Client(transport, read_timeout_seconds=5) as client:
                listing = await client.list_tools()
                assert {tool.name for tool in listing.tools} == {"search", "get_document", "status"}
                result = await client.call_tool("status", {})
                assert not result.is_error
                assert result.structured_content["documents"] == 0


async def test_http_rejects_foreign_host_and_origin_even_with_bearer(kb):
    token = "a" * 64
    app = create_http_app(kb, token)
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    async with app.app.router.lifespan_context(app.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            host = await client.post("http://attacker.invalid/mcp", headers=headers, json={})
            assert host.status_code == 421
            origin = await client.post(
                "http://localhost:8080/mcp",
                headers={**headers, "Origin": "https://attacker.invalid"},
                json={},
            )
            assert origin.status_code == 403
