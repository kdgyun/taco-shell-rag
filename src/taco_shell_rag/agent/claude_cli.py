import json
import os
import selectors
import signal
import subprocess
import tempfile
from pathlib import Path
from time import monotonic

from .models import GenerationConfig
from .runner import SYSTEM


def run_claude(question: str, config: GenerationConfig):
    if not question.strip() or len(question) > 8000:
        raise ValueError("question must contain 1..8000 characters")
    if config.mode != "agent":
        raise ValueError("Claude CLI supports agent mode only; use an HTTP provider for RAG mode")
    import httpx

    url = httpx.URL(config.mcp_url)
    if url.scheme not in {"http", "https"} or url.username or url.password:
        raise ValueError("Claude CLI requires a configured HTTP MCP URL")
    token = os.environ.get(config.mcp_token_env, "")
    if len(token) < 32:
        raise ValueError("Claude CLI requires the MCP bearer token in its configured environment")
    mcp = {
        "mcpServers": {
            "taco": {
                "type": "http",
                "url": config.mcp_url,
                "headers": {"Authorization": "Bearer " + token},
            }
        }
    }
    with tempfile.TemporaryDirectory(prefix="taco-agent-") as work:
        path = Path(work) / "mcp.json"
        path.write_text(json.dumps(mcp))
        path.chmod(0o600)
        args = [
            config.cli_bin,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            config.model,
            "--max-turns",
            str(config.max_steps),
            "--mcp-config",
            str(path),
            "--strict-mcp-config",
            "--tools",
            "",
            "--allowedTools",
            "mcp__taco__search mcp__taco__get_document mcp__taco__status",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--no-chrome",
            "--disable-slash-commands",
            "--setting-sources",
            "",
            "--settings",
            '{"disableAllHooks":true,"enabledPlugins":{}}',
            "--system-prompt",
            SYSTEM,
        ]
        process = subprocess.Popen(
            args,
            cwd=work,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        trace, total, pending, final = [], 0, bytearray(), None
        deadline = monotonic() + config.timeout

        def consume(line):
            nonlocal final
            if not line.strip():
                return
            event = json.loads(line)
            if event.get("type") == "result":
                final = event
            for content in event.get("message", {}).get("content", []):
                if content.get("type") == "tool_use":
                    name = content.get("name", "")
                    if name not in {
                        "mcp__taco__search",
                        "mcp__taco__get_document",
                        "mcp__taco__status",
                    }:
                        raise ValueError("Claude attempted an unexpected tool")
                    trace.append({"tool": name, "id": content.get("id")})

        try:
            process.stdin.write(question.encode())
            process.stdin.close()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if monotonic() >= deadline:
                        raise TimeoutError("Claude CLI deadline exceeded")
                    for key, _ in selector.select(timeout=max(0, min(1, deadline - monotonic()))):
                        block = os.read(key.fileobj.fileno(), 65536)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        total += len(block)
                        if total > 4_000_000:
                            raise ValueError("Claude CLI output exceeded the configured bound")
                        pending.extend(block)
                        while b"\n" in pending:
                            line, _, pending = pending.partition(b"\n")
                            consume(line)
            # 프로세스가 마지막 줄의 개행 없이 종료해도 완료 이벤트는 처리한다.
            # Process the final event even when the stream ends without a trailing newline.
            consume(pending)
            process.wait(timeout=max(0.1, deadline - monotonic()))
            if (
                process.returncode
                or not final
                or final.get("is_error")
                or final.get("subtype") != "success"
            ):
                raise RuntimeError("Claude CLI did not complete; check its own authentication")
            return {
                "status": "completed",
                "answer": final.get("result", ""),
                "provider": "claude-cli",
                "requested_model": config.model,
                "served_model": None,
                "trace": trace,
                "tool_calls": len(trace),
                "usage": final.get("usage"),
                "limitations": "CLI-managed agent; served model and evidence completeness "
                "are not established by this trace.",
            }
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            process.stdout.close()
