import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from qdrant_client import models

from taco_shell_rag.core.contracts import Document, Snapshot

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "contract-test-only-" + "x" * 48


@pytest.fixture(scope="session")
def go_binaries(tmp_path_factory):
    if not shutil.which("go"):
        pytest.skip("Go toolchain required for cross-language MCP integration")
    directory = tmp_path_factory.mktemp("go-mcp-binaries")
    binaries = {}
    for name in ["http", "stdio"]:
        binary = directory / f"mcp-{name}"
        subprocess.run(
            ["go", "build", "-mod=readonly", "-o", str(binary), f"./cmd/mcp-{name}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=180,
        )
        binaries[name] = str(binary)
    return binaries


@pytest.fixture
def go_stack(kb, tmp_path, go_binaries):
    class Notes:
        id = "notes"

        def __init__(self):
            self.documents = [
                Document(
                    "policy", self.id, "ABC-123 refund policy", title="Refund", path="refund.md"
                )
            ]

        def scan(self, checkpoint):
            return Snapshot(self.documents, "snapshot")

    notes = Notes()
    kb.sources = {notes.id: notes}
    assert kb.sync()[notes.id]["status"] == "ok"
    calls = {"embed": 0, "rerank": 0}
    control = {
        "fail_models": False,
        "change_execution": False,
        "hide_dense": False,
        "echo_stale_fingerprint": False,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if control["fail_models"]:
                self.reply({"error": "private-upstream-body"}, 503)
            else:
                space = kb.embedder.space()
                if control["echo_stale_fingerprint"]:
                    space = replace(space, model="changed-with-stale-fingerprint")
                self.reply(
                    {
                        "ready": True,
                        "embedding_space": asdict(space),
                        "embedding_fingerprint": kb.embedder.space().fingerprint,
                    }
                )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/points/query"):
                points = (
                    []
                    if control["hide_dense"]
                    else kb.index.client.query_points(
                        "test",
                        query=body["query"],
                        query_filter=models.Filter(**body["filter"]),
                        limit=body["limit"],
                        with_payload=True,
                    ).points
                )
                self.reply({"result": {"points": [p.model_dump() for p in points]}})
                return
            if control["fail_models"]:
                self.reply({"error": "private-upstream-body"}, 503)
                return
            if self.path == "/embed":
                calls["embed"] += 1
                result = kb.embedder.embed(body["texts"], body["purpose"])
                space = (
                    replace(result.space, model="changed")
                    if control["change_execution"]
                    else result.space
                )
                self.reply(
                    {
                        "vectors": result.vectors,
                        "embedding_space": asdict(space),
                        "embedding_fingerprint": space.fingerprint,
                    }
                )
            elif self.path == "/rerank":
                calls["rerank"] += 1
                result = kb.reranker.rerank(body["query"], body["texts"])
                self.reply({"scores": result.scores, "reranker": asdict(result.info)})
            else:
                self.reply({}, 404)

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    backend_url = f"http://127.0.0.1:{backend.server_port}"
    config = tmp_path / "go.toml"
    config.write_text(
        f'embedding_url = "{backend_url}"\n'
        f'qdrant_url = "{backend_url}"\n'
        'collection = "test"\n'
        f"catalog = {json.dumps(str(kb.ledger.path))}\n"
        'host = "127.0.0.1"\n'
        f"port = {port}\n"
        '[[sources]]\nid = "notes"\ntype = "folder"\npath = "."\n'
    )
    process = subprocess.Popen(
        [go_binaries["http"], "--config", str(config)],
        env={**os.environ, "MCP_AUTH_TOKEN": TOKEN},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(trust_env=False, timeout=1) as http:
            deadline = time.monotonic() + 10
            while True:
                if process.poll() is not None:
                    pytest.fail(
                        "Go HTTP MCP exited before readiness: " + process.stderr.read().decode()
                    )
                try:
                    if http.get(url + "/healthz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    pytest.fail("Go HTTP MCP did not become ready")
                time.sleep(0.02)
        yield SimpleNamespace(
            kb=kb,
            notes=notes,
            calls=calls,
            control=control,
            url=url,
            config=config,
            binaries=go_binaries,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stderr.close()
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)


async def test_actual_go_stdio_reads_python_publication_and_all_search_modes(go_stack):
    stack = go_stack
    parameters = StdioServerParameters(
        command=stack.binaries["stdio"], args=["--config", str(stack.config)]
    )
    async with Client(parameters, read_timeout_seconds=10) as client:
        listing = await client.list_tools()
        assert {tool.name for tool in listing.tools} == {
            "search",
            "keyword_search",
            "deep_search",
            "get_document",
            "status",
        }
        for tool in ["search", "keyword_search", "deep_search"]:
            result = await client.call_tool(tool, {"query": "ABC-123 refund"})
            assert not result.is_error, result.content
            evidence = result.structured_content
            assert evidence["results"][0]["chunk"]["text"] == "ABC-123 refund policy"
            assert "ABC-123 refund policy" in str(result.content)
        assert stack.calls["embed"] == 2
        assert stack.calls["rerank"] == 2
        chunk = evidence["results"][0]["chunk"]
        read = await client.call_tool(
            "get_document",
            {
                "document_key": chunk["document_key"],
                "version": chunk["version"],
            },
        )
        assert not read.is_error
        assert read.structured_content["chunks"][0]["text"] == "ABC-123 refund policy"
        status = await client.call_tool("status", {})
        assert status.structured_content["implementation"] == "go"


async def test_actual_go_http_auth_keyword_outage_and_snapshot_refresh(go_stack):
    stack = go_stack
    with httpx.Client(trust_env=False) as http:
        assert http.post(stack.url + "/mcp", json={}).status_code == 401
        assert (
            http.post(
                stack.url + "/mcp",
                json={},
                headers={"Authorization": "Bearer " + TOKEN, "Origin": "https://attacker.invalid"},
            ).status_code
            == 403
        )
    async with httpx2.AsyncClient(headers={"Authorization": "Bearer " + TOKEN}) as http:
        transport = streamable_http_client(stack.url + "/mcp", http_client=http)
        async with Client(transport, read_timeout_seconds=10) as client:
            stack.control["fail_models"] = True
            keyword = await client.call_tool("keyword_search", {"query": "ABC-123"})
            assert not keyword.is_error
            assert len(keyword.structured_content["results"]) == 1
            failed = await client.call_tool("search", {"query": "refund"})
            assert failed.is_error
            assert "private-upstream" not in str(failed.content)
            old = keyword.structured_content["results"][0]["chunk"]
            stack.control["fail_models"] = False
            stack.notes.documents = [replace(stack.notes.documents[0], text="ABC-456 replacement")]
            assert stack.kb.sync()["notes"]["status"] == "ok"
            stale = await client.call_tool(
                "get_document",
                {
                    "document_key": old["document_key"],
                    "version": old["version"],
                },
            )
            assert stale.is_error
            old_keyword = await client.call_tool("keyword_search", {"query": "ABC-123"})
            assert old_keyword.structured_content["results"] == []
            fresh = await client.call_tool("keyword_search", {"query": "ABC-456"})
            assert fresh.structured_content["results"][0]["chunk"]["text"] == "ABC-456 replacement"


async def test_go_hybrid_recovers_lexical_only_hit_and_rejects_execution_change(go_stack):
    stack = go_stack
    async with Client(
        StdioServerParameters(command=stack.binaries["stdio"], args=["--config", str(stack.config)])
    ) as client:
        stack.control["hide_dense"] = True
        result = await client.call_tool("search", {"query": "ABC-123"})
        assert not result.is_error
        assert result.structured_content["results"][0]["matched_by"] == ["keyword"]
        stack.control["change_execution"] = True
        failed = await client.call_tool("search", {"query": "ABC-123"})
        assert failed.is_error
        assert "execution space changed" in str(failed.content)


async def test_go_rejects_stale_fingerprint_with_changed_health_identity(go_stack):
    stack = go_stack
    stack.control["echo_stale_fingerprint"] = True
    async with Client(
        StdioServerParameters(command=stack.binaries["stdio"], args=["--config", str(stack.config)])
    ) as client:
        failed = await client.call_tool("search", {"query": "ABC-123"})
        assert failed.is_error
        assert "indexed identity" in str(failed.content)
        assert stack.calls["embed"] == 0


def test_python_cli_uses_go_for_each_search_mode(go_stack, monkeypatch, capsys):
    from taco_shell_rag import cli

    stack = go_stack
    monkeypatch.setenv("MCP_STDIO_BIN", stack.binaries["stdio"])

    def unexpected(*_):
        raise AssertionError("search must not initialize a Python indexer or MCP server")

    monkeypatch.setattr(cli, "build_knowledge_base", unexpected)
    for mode in ["keyword", "hybrid", "dense", "planned"]:
        monkeypatch.setattr(
            "sys.argv",
            [
                "taco",
                "--config",
                str(stack.config),
                "search",
                "ABC-123",
                "--mode",
                mode,
            ],
        )
        assert cli.main() == 0
        result = json.loads(capsys.readouterr().out)
        assert result["mode"] == mode
        assert result["results"][0]["chunk"]["text"] == "ABC-123 refund policy"
    assert stack.calls == {"embed": 3, "rerank": 3}
