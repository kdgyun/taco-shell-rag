import argparse
import asyncio
import json
import os
import sys
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

from .config import build_knowledge_base, load_settings
from .core.contracts import EmbeddingSpaceMismatch
from .core.service import require_gpu, require_reranker
from .model_config import ModelConfigurationError, load_model_settings, model_catalog


def main():
    parser = argparse.ArgumentParser(prog="taco")
    parser.add_argument("--config", type=Path, default=Path("taco.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync")
    sync.add_argument("--watch", action="store_true")
    sync.add_argument("--interval", type=int, default=180)
    sub.add_parser("status")
    sub.add_parser("doctor")
    model_options = sub.add_parser("models")
    model_options.add_argument("--model-config", type=Path)
    search = sub.add_parser("search")
    search.add_argument("query")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--model-config", type=Path)
    server = sub.add_parser("serve")
    server.add_argument("--transport", choices=["http", "stdio"], default="http")
    args = parser.parse_args()
    try:
        if args.command == "models":
            result = (
                load_model_settings(args.model_config).model_dump()
                if args.model_config
                else model_catalog()
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        settings = load_settings(args.config)
        if args.command == "ask":
            if args.model_config:
                from .agent.models import GenerationConfig

                with args.model_config.open("rb") as stream:
                    settings.generation = GenerationConfig.model_validate(tomllib.load(stream))
            if settings.generation is None:
                raise ValueError("configure a generation provider before using ask")
            if settings.generation.provider == "claude-cli":
                from .agent.claude_cli import run_claude

                # 원격 MCP를 쓰는 CLI 실행에는 로컬 색인이나 저장소 연결이 필요 없다.
                # A CLI using remote MCP does not need a local catalog or vector connection.
                answer = run_claude(args.question, settings.generation)
                print(json.dumps(answer, ensure_ascii=False, indent=2))
                return int(answer["status"] != "completed")
        kb = build_knowledge_base(settings, args.config.resolve().parent)
        if args.command == "status":
            print(json.dumps(kb.ledger.status(), indent=2))
        elif args.command == "doctor":
            space = kb.embedder.space()
            require_gpu(space)
            kb.ledger.check_space(space.fingerprint)
            vector_index = kb.index.inspect(space.dimension)
            reranker = None
            if kb.rerank_enabled:
                reranker = kb.reranker.describe()
                require_reranker(reranker)
            print(
                json.dumps(
                    {
                        "ready": True,
                        "embedding_space": asdict(space),
                        "vector_index": vector_index,
                        "reranker": asdict(reranker) if reranker else None,
                        "sources": sorted(kb.sources),
                    }
                )
            )
        elif args.command == "search":
            print(json.dumps(kb.search(args.query), ensure_ascii=False, indent=2))
        elif args.command == "sync":
            if args.interval < 10:
                raise ValueError("watch interval must be at least 10 seconds")
            while True:
                try:
                    results = kb.sync()
                except Exception as error:
                    if not args.watch:
                        raise
                    # 추론 서비스가 잠시 중단돼도 감시 루프는 다음 주기에 다시 시도한다.
                    # A temporary inference outage must not terminate the watch loop.
                    results = {"sync": {"status": "error", "error": type(error).__name__}}
                print(json.dumps(results), flush=True)
                if not args.watch:
                    return int(any(result["status"] != "ok" for result in results.values()))
                time.sleep(args.interval)
        elif args.command == "ask":
            from mcp import Client

            from .agent.models import HttpModel
            from .agent.runner import run_agent, run_rag
            from .mcp_server import create_server

            async def answer_question():
                model = HttpModel(settings.generation)
                try:
                    async with Client(create_server(kb)) as client:
                        if settings.generation.mode == "rag":
                            return await run_rag(args.question, model, client)
                        return await run_agent(
                            args.question, model, client, settings.generation.max_steps
                        )
                finally:
                    await model.client.aclose()

            answer = asyncio.run(answer_question())
            print(json.dumps(answer, ensure_ascii=False, indent=2))
            return int(answer["status"] != "completed")
        else:
            from .mcp_server import create_http_app, create_server

            if args.transport == "stdio":
                create_server(kb).run(transport="stdio")
            else:
                import uvicorn

                app = create_http_app(
                    kb,
                    os.environ.get(settings.token_env, ""),
                    settings.allowed_hosts,
                    settings.allowed_origins,
                )
                uvicorn.run(app, host=settings.host, port=settings.port)
        return 0
    except Exception as error:
        # 사용자에게 보여 주는 오류에 토큰이나 외부 응답 본문이 섞이지 않게 한다.
        # Keep tokens and external response bodies out of public diagnostics.
        hint = "check configuration and service health"
        if isinstance(error, ModelConfigurationError):
            hint = str(error)
        elif isinstance(error, EmbeddingSpaceMismatch):
            hint = (
                "embedding space changed; check serving model consistency; if the change is "
                "intended, select a new collection and catalog, then reindex"
            )
        elif isinstance(error, PermissionError):
            hint = (
                "check config/source read permissions and catalog write permissions "
                "(container UID 10001); see docs/install.md"
            )
        print(f"taco: {type(error).__name__}; {hint}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
