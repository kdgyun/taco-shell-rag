import os
import re
import sys
from collections.abc import Mapping


def server_command(env: Mapping[str, str]) -> list[str]:
    model = env.get("GENERATOR_MODEL", "").strip()
    revision = env.get("GENERATOR_REVISION", "")
    if not model or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("set GENERATOR_MODEL and an immutable 40-character GENERATOR_REVISION")
    if not env.get("VLLM_API_KEY"):
        raise ValueError("generator API credential is unset")
    try:
        context = int(env.get("GENERATOR_CONTEXT", "8192"))
        memory = float(env.get("GENERATOR_MEMORY_FRACTION", "0.7"))
        sequences = (
            int(env["GENERATOR_MAX_SEQUENCES"]) if env.get("GENERATOR_MAX_SEQUENCES") else None
        )
    except ValueError:
        raise ValueError("invalid generator context, memory fraction or sequence count") from None
    if context < 256 or not 0 < memory <= 1 or (sequences is not None and sequences < 1):
        raise ValueError("generator context, memory fraction or sequence count is out of range")
    args = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--revision",
        revision,
        "--served-model-name",
        model,
        "--max-model-len",
        str(context),
        "--gpu-memory-utilization",
        str(memory),
    ]
    tool_parser = env.get("GENERATOR_TOOL_CALL_PARSER", "")
    if tool_parser:
        args.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])
    if parser := env.get("GENERATOR_REASONING_PARSER", ""):
        args.extend(["--reasoning-parser", parser])
    if sequences is not None:
        args.extend(["--max-num-seqs", str(sequences)])
    eager = env.get("GENERATOR_EAGER", "false").lower()
    if eager not in {"true", "false", "1", "0"}:
        raise ValueError("GENERATOR_EAGER must be true or false")
    if eager in {"true", "1"}:
        args.append("--enforce-eager")
    return args


def main():
    try:
        command = server_command(os.environ)
    except ValueError as error:
        print(f"generator: {error}", file=sys.stderr)
        return 1
    # 키를 명령행에 넣지 않는다. 이미지가 읽는 환경변수로만 전달한다.
    # Keep credentials in the runtime environment, never in process arguments.
    os.execvp(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())
