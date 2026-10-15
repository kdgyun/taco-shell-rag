import importlib.util
import stat
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.specifiers import SpecifierSet

from taco_shell_rag.config import load_settings


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


def test_host_check_reports_missing_buildx_without_installing(monkeypatch):
    spec = importlib.util.spec_from_file_location("bootstrap", Path("scripts/bootstrap.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/test/bin/tool")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=int(command == ["docker", "buildx", "version"]))

    monkeypatch.setattr(module.subprocess, "run", run)
    checks = module.check_host()
    assert not checks["buildx"]
    assert all(value for key, value in checks.items() if key != "buildx")
    assert ["docker", "buildx", "version"] in commands


def test_example_config_and_matching_install_skills():
    config = load_settings(Path("taco.toml"))
    assert config.token_env == "MCP_AUTH_TOKEN"
    assert config.sources[0].id == "handbook"
    assert (
        Path(".agents/skills/taco-install/SKILL.md").read_bytes()
        == Path(".claude/skills/taco-install/SKILL.md").read_bytes()
    )


@pytest.mark.parametrize("filename", ["Dockerfile", "Dockerfile.gpu"])
def test_docker_dependencies_are_cached_separately_from_source(filename):
    dockerfile = Path(filename).read_text()
    assert dockerfile.index("--no-install-project") < dockerfile.index("COPY src ./src")
    assert "--mount=type=cache,target=/root/.cache/uv" in dockerfile
    assert "--link-mode=copy" in dockerfile


@pytest.mark.parametrize("filename", ["Dockerfile", "Dockerfile.gpu"])
def test_docker_project_build_copies_readme_and_license_metadata(filename):
    dockerfile = Path(filename).read_text()
    assert dockerfile.index("COPY README.md LICENSE ./") < dockerfile.index("--no-editable")


def test_python_support_matches_the_pinned_cuda_wheel_versions():
    with Path("pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    supported = SpecifierSet(project["project"]["requires-python"])
    assert all(version in supported for version in ["3.11", "3.12", "3.13"])
    assert all(version not in supported for version in ["3.10", "3.14"])
