import os
import shutil
from pathlib import Path

from mcp.client.stdio import StdioServerParameters


class MCPBinaryUnavailable(ValueError):
    pass


def server_binary(transport: str) -> str:
    if transport not in {"http", "stdio"}:
        raise ValueError("unsupported MCP transport")
    override = os.environ.get(f"MCP_{transport.upper()}_BIN")
    executable = override or shutil.which(f"mcp-{transport}")
    if not executable:
        # wheel에는 플랫폼별 Go 바이너리를 넣지 않는다. 소스 설치는 make build로 준비한다.
        # Wheels do not bundle platform-specific Go binaries; source installs use make build.
        root = Path(__file__).resolve().parents[2]
        candidate = root / "bin" / f"mcp-{transport}"
        if candidate.is_file():
            executable = str(candidate)
    if not executable or not os.access(executable, os.X_OK):
        raise MCPBinaryUnavailable(
            f"Go MCP binary is missing; run make build or configure MCP_{transport.upper()}_BIN"
        )
    return executable


def stdio_parameters(config: Path, settings) -> StdioServerParameters:
    credentials = [
        settings.embedding_token_env,
        settings.reranker_token_env,
        settings.qdrant_token_env,
    ]
    # 하위 프로세스에는 설정에서 지정한 자격 증명만 전달한다. 전체 환경을 복사하지 않는다.
    # Pass only explicitly configured credentials, not the parent's entire environment.
    env = {name: os.environ[name] for name in credentials if name and name in os.environ}
    return StdioServerParameters(
        command=server_binary("stdio"), args=["--config", str(config.resolve())], env=env
    )
