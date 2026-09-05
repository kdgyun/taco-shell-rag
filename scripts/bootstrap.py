"""GPU 호스트를 점검한다. 명시적으로 요청한 경우에만 비공개 설정 파일을 만든다.

Check the GPU host and create a private configuration file on explicit request.
"""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path


def create_env(root: Path) -> bool:
    target = root / ".env"
    if target.exists():
        return False
    template = (root / ".env.example").read_text()
    content = template.replace(
        "MCP_AUTH_TOKEN=\n", "MCP_AUTH_TOKEN=" + secrets.token_hex(32) + "\n"
    )
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content)
    return True


def check_host() -> dict:
    checks = {"linux": sys.platform == "linux", "docker": False, "compose": False, "gpu": False}
    commands = {
        "docker": ["docker", "info"],
        "compose": ["docker", "compose", "version"],
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    }
    for name, command in commands.items():
        if shutil.which(command[0]):
            try:
                result = subprocess.run(command, capture_output=True, timeout=15, check=False)
                checks[name] = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                pass
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Read-only host checks")
    parser.add_argument("--write-env", action="store_true", help="Create .env; never overwrite it")
    args = parser.parse_args()
    if args.write_env:
        created = create_env(Path(__file__).resolve().parents[1])
        print("Created private .env" if created else "Kept existing .env")
    if args.check or not args.write_env:
        checks = check_host()
        print(json.dumps(checks))
        if not all(checks.values()):
            print("GPU host is not ready. See docs/install.md; CPU fallback is not supported.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
