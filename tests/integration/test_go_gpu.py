import json
import os
import secrets
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from taco_shell_rag.config import build_knowledge_base, load_settings
from taco_shell_rag.indexer.pipeline import require_gpu, require_reranker

# 실행 중인 GPU·Qdrant에 접속하는 검사는 별도로 요청해야 한다.
# Live GPU and Qdrant access requires a separate, explicit opt-in.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("RUN_GO_GPU_TESTS") != "1", reason="requires RUN_GO_GPU_TESTS=1"
    ),
]


@pytest.fixture
def live_stack(tmp_path, go_binaries):
    required = ["GPU_TEST_INFERENCE_URL", "GPU_TEST_QDRANT_URL"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail("Set explicit test endpoints: " + ", ".join(missing))

    handbook = tmp_path / "handbook"
    operations = tmp_path / "operations"
    handbook.mkdir()
    operations.mkdir()
    refund = handbook / "refund.md"
    refund.write_text(
        "QA-101 refund policy: unused subscriptions are refundable within seven days."
    )
    (handbook / "other.md").write_text("QA-1010 is a separate policy for contact details.")
    (operations / "billing.md").write_text(
        "QA-202 billing/refund.go handles refund requests. 환불 처리는 결제 서비스에서 수행한다."
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    values = {
        "embedding_url": os.environ["GPU_TEST_INFERENCE_URL"],
        "reranker_url": os.environ.get("GPU_TEST_RERANKER_URL")
        or os.environ["GPU_TEST_INFERENCE_URL"],
        "qdrant_url": os.environ["GPU_TEST_QDRANT_URL"],
        "collection": "gpu_test_" + uuid4().hex,
        "catalog": str(tmp_path / "catalog.sqlite3"),
        "host": "127.0.0.1",
        "port": port,
    }
    for role in ["embedding", "reranker", "qdrant"]:
        name = f"GPU_TEST_{role.upper()}_TOKEN"
        if os.environ.get(name):
            values[f"{role}_token_env"] = name
    config = tmp_path / "mcp.toml"
    config.write_text(
        "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items())
        + '[[sources]]\nid = "handbook"\ntype = "folder"\npath = "handbook"\n'
        + '[[sources]]\nid = "operations"\ntype = "folder"\npath = "operations"\n'
    )
    settings = load_settings(config)
    kb = build_knowledge_base(settings, tmp_path)
    owned = False
    try:
        space = kb.embedder.space()
        require_gpu(space)
        reranker = kb.reranker.describe()
        require_reranker(reranker)
        assert not kb.index.client.collection_exists(settings.collection)
        # 임의 이름의 전용 컬렉션만 만들고 정리한다. 기존 색인은 건드리지 않는다.
        # Create and clean up only this uniquely named collection, never an existing index.
        owned = True
        first = kb.sync()
        assert {key: value["status"] for key, value in first.items()} == {
            "handbook": "ok",
            "operations": "ok",
        }, first
        assert first["handbook"]["changed"] == 2
        assert first["operations"]["changed"] == 1
        assert all(value["changed"] == 0 for value in kb.sync().values())
        yield SimpleNamespace(
            kb=kb,
            settings=settings,
            config=config,
            binaries=go_binaries,
            port=port,
            refund=refund,
            space=space,
            reranker=reranker,
        )
    finally:
        try:
            if owned and kb.index.client.collection_exists(settings.collection):
                kb.index.client.delete_collection(settings.collection)
        finally:
            kb.index.client.close()
            kb.embedder.client.close()
            kb.reranker.client.close()


# MCP의 취소 범위는 같은 태스크 안에서 열고 닫아야 한다.
# Enter and exit MCP cancellation scopes within the same task.
@asynccontextmanager
async def connect_client(stack, transport):
    if transport == "stdio":
        parameters = StdioServerParameters(
            command=stack.binaries["stdio"], args=["--config", str(stack.config)]
        )
        async with Client(parameters, read_timeout_seconds=120) as client:
            yield client
        return

    token = secrets.token_hex(32)
    process = subprocess.Popen(
        [stack.binaries["http"], "--config", str(stack.config)],
        env={**os.environ, "MCP_AUTH_TOKEN": token},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{stack.port}"
    try:
        with httpx.Client(trust_env=False, timeout=1) as http:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    pytest.fail("Go HTTP MCP exited: " + process.stderr.read().decode())
                try:
                    if http.get(url + "/healthz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() >= deadline:
                    pytest.fail("Go HTTP MCP did not become ready")
                time.sleep(0.05)
            assert http.post(url + "/mcp", json={}).status_code == 401
        async with httpx2.AsyncClient(headers={"Authorization": "Bearer " + token}) as http:
            async with Client(
                streamable_http_client(url + "/mcp", http_client=http), read_timeout_seconds=120
            ) as client:
                yield client
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stderr.close()


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_go_mcp_with_real_gpu_models_and_qdrant(live_stack, transport):
    async with connect_client(live_stack, transport) as client:
        await verify_tools(live_stack, client)


async def verify_tools(stack, client):
    listing = await client.list_tools()
    assert {tool.name for tool in listing.tools} == {
        "search", "keyword_search", "deep_search", "get_document", "status"
    }
    for tool, extra in [("search", {}), ("search", {"mode": "dense"})]:
        result = await client.call_tool(tool, {"query": "refund policy", **extra})
        assert not result.is_error, result.content
        data = result.structured_content
        assert data["embedding_space"] == asdict(stack.space)
        assert data["reranker"] == asdict(stack.reranker)
        assert data["score_kind"] == "reranker_raw_uncalibrated"
        assert data["results"]
        assert not data["absence_claim_allowed"]
        assert all(hit["chunk"]["version"] for hit in data["results"])

    keyword = await client.call_tool("keyword_search", {"query": "QA-101"})
    assert not keyword.is_error, keyword.content
    assert len(keyword.structured_content["results"]) == 1
    assert keyword.structured_content["score_kind"] == "bm25"
    old = keyword.structured_content["results"][0]["chunk"]
    original = await client.call_tool(
        "get_document", {"document_key": old["document_key"], "version": old["version"]}
    )
    assert not original.is_error, original.content
    assert "seven days" in original.structured_content["chunks"][0]["text"]

    planned = await client.call_tool(
        "deep_search", {"query": "QA-101 refund billing/refund.go MISSING-999", "limit": 10}
    )
    assert not planned.is_error, planned.content
    data = planned.structured_content
    assert data["mode"] == "planned"
    coverage = {
        item["term"]: item["found_in_returned_evidence"] for item in data["anchor_coverage"]
    }
    assert coverage["QA-101"]
    assert not coverage["MISSING-999"]
    assert data["coverage_gaps"]
    scoped = await client.call_tool("search", {"query": "환불", "source_id": "operations"})
    assert not scoped.is_error, scoped.content
    assert scoped.structured_content["results"]
    assert all(
        hit["chunk"]["source_id"] == "operations"
        for hit in scoped.structured_content["results"]
    )

    # 연결을 유지한 채 게시를 바꿔 WAL 갱신과 오래된 버전 거부를 함께 확인한다.
    # Keep the MCP session open while publishing to verify WAL refresh and stale-version rejection.
    stack.refund.write_text(
        "QA-303 replaces the refund policy: unused subscriptions allow 14 days."
    )
    updated = stack.kb.sync()
    assert updated["handbook"]["status"] == "ok", updated
    assert updated["handbook"]["changed"] == 1
    stale = await client.call_tool(
        "get_document", {"document_key": old["document_key"], "version": old["version"]}
    )
    assert stale.is_error
    previous = await client.call_tool("keyword_search", {"query": "QA-101"})
    assert not previous.is_error and previous.structured_content["results"] == []
    fresh = await client.call_tool("search", {"query": "QA-303 refund"})
    assert not fresh.is_error, fresh.content
    assert any("QA-303" in hit["chunk"]["text"] for hit in fresh.structured_content["results"])
    status = await client.call_tool("status", {})
    assert not status.is_error, status.content
    assert status.structured_content["implementation"] == "go"
