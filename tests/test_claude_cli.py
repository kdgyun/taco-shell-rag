import json
import sys

from taco_shell_rag.agent.claude_cli import run_claude
from taco_shell_rag.agent.models import GenerationConfig


def test_claude_cli_uses_only_configured_read_mcp_and_private_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "x" * 64)
    script = tmp_path / "fake-claude"
    script.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json,sys,os
from pathlib import Path
args=sys.argv[1:]
assert args[args.index("--tools")+1] == ""
assert "--strict-mcp-config" in args
assert "--no-session-persistence" in args
assert args[args.index("--allowedTools")+1].split() == [
    "mcp__taco__search", "mcp__taco__get_document", "mcp__taco__status"
]
path=Path(args[args.index("--mcp-config")+1])
assert path.stat().st_mode & 0o777 == 0o600
config=json.loads(path.read_text())
assert config["mcpServers"]["taco"]["url"] == "http://127.0.0.1:8080/mcp"
assert sys.stdin.read() == "refund?"
print(json.dumps({"type":"assistant","message":{"content":[{"type":"tool_use","id":"a","name":"mcp__taco__search"}]}}))
print(json.dumps({"type":"result","subtype":"success","is_error":False,"result":"Answer","usage":{"input_tokens":10}}))
"""
    )
    script.chmod(0o700)
    config = GenerationConfig(provider="claude-cli", model="test", cli_bin=str(script))
    result = run_claude("refund?", config)
    assert result["answer"] == "Answer"
    assert result["tool_calls"] == 1
    assert "x" * 64 not in json.dumps(result)
