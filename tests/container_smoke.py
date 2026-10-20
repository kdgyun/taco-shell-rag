"""Exercise the installed Linux Python package and Go stdio binary without model calls."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from taco_shell_rag.core.contracts import Document
from taco_shell_rag.indexer.catalog import Catalog
from taco_shell_rag.indexer.chunking import split_document


async def check(root: Path):
    ledger = Catalog(root / "catalog.sqlite3")
    document = Document("example", "notes", "ABC-123 refund policy")
    ledger.activate(document, split_document(document, 64, 8))
    config = root / "rag.toml"
    config.write_text(
        'catalog = "catalog.sqlite3"\n[[sources]]\nid = "notes"\ntype = "folder"\npath = "."\n'
    )
    # 실제 WAL 쓰기 연결을 유지한 채 다른 언어의 읽기 전용 연결을 확인한다.
    # Keep a real WAL writer open while testing the other language's read-only connection.
    with ledger.connect() as writer:
        writer.execute("INSERT INTO settings VALUES ('smoke','1')")
        writer.commit()
        async with Client(
            StdioServerParameters(command="mcp-stdio", args=["--config", str(config)])
        ) as client:
            keyword = await client.call_tool("keyword_search", {"query": "ABC-123"})
            assert not keyword.is_error, keyword.content
            hit = keyword.structured_content["results"][0]["chunk"]
            assert hit["text"] == document.text
            read = await client.call_tool(
                "get_document",
                {
                    "document_key": hit["document_key"],
                    "version": hit["version"],
                },
            )
            assert not read.is_error
            assert read.structured_content["chunks"][0]["text"] == document.text
            status = await client.call_tool("status", {})
            assert status.structured_content["implementation"] == "go"
            assert status.structured_content["documents"] == 1
    print(
        json.dumps(
            {
                "ok": True,
                "uid": os.getuid(),
                "scope": "Linux Go/Python WAL and MCP; no model inference",
            }
        )
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as work:
        asyncio.run(check(Path(work)))
