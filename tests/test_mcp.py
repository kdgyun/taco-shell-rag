from mcp import Client

from taco_shell_rag.core.contracts import Document, Snapshot
from taco_shell_rag.mcp_server import create_server


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
