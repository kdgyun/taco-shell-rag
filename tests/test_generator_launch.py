import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def launch():
    spec = importlib.util.spec_from_file_location(
        "serve_generator", Path("scripts/serve_generator.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings():
    return {
        "GENERATOR_MODEL": "custom/model",
        "GENERATOR_REVISION": "a" * 40,
        "VLLM_API_KEY": "private-test-secret",
    }


def test_text_only_server_does_not_implicitly_enable_tools(launch, settings):
    command = launch.server_command(settings)
    assert "custom/model" in command
    assert "--tool-call-parser" not in command
    assert "--enable-auto-tool-choice" not in command
    assert "--reasoning-parser" not in command
    assert "--enforce-eager" not in command
    assert "private-test-secret" not in command


def test_model_specific_server_options_are_forwarded(launch, settings):
    settings.update(
        GENERATOR_TOOL_CALL_PARSER="hermes",
        GENERATOR_REASONING_PARSER="qwen3",
        GENERATOR_MAX_SEQUENCES="1",
        GENERATOR_EAGER="true",
        GENERATOR_CONTEXT="4096",
        GENERATOR_MEMORY_FRACTION="0.55",
    )
    command = launch.server_command(settings)
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.55"
    assert "--enforce-eager" in command


@pytest.mark.parametrize(
    "key,value",
    [
        ("GENERATOR_MODEL", ""),
        ("GENERATOR_REVISION", "main"),
        ("VLLM_API_KEY", ""),
        ("GENERATOR_CONTEXT", "0"),
        ("GENERATOR_MEMORY_FRACTION", "nan"),
        ("GENERATOR_MEMORY_FRACTION", "1.1"),
        ("GENERATOR_MAX_SEQUENCES", "0"),
        ("GENERATOR_CONTEXT", "private-bad-input"),
        ("GENERATOR_EAGER", "private-bad-input"),
    ],
)
def test_bad_settings_fail_without_echoing_inputs(launch, settings, key, value):
    settings[key] = value
    with pytest.raises(ValueError) as error:
        launch.server_command(settings)
    assert "private-bad-input" not in str(error.value)
