import importlib.util
import stat
from pathlib import Path


def test_bootstrap_creates_private_env_once_without_overwriting(tmp_path):
    spec = importlib.util.spec_from_file_location("bootstrap", Path("scripts/bootstrap.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / ".env.example").write_text("# 설명 / description\nMCP_AUTH_TOKEN=\n")
    assert module.create_env(tmp_path)
    env = tmp_path / ".env"
    original = env.read_text()
    assert len(original.split("MCP_AUTH_TOKEN=")[1].strip()) == 64
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert not module.create_env(tmp_path)
    assert env.read_text() == original
