import importlib.util
import io
import stat
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

LICENSE = b"test license text\n"
METADATA = b"Metadata-Version: 2.4\nLicense-Expression: Apache-2.0\nLicense-File: LICENSE\n\n"


@pytest.fixture
def release():
    spec = importlib.util.spec_from_file_location("check_release", Path("scripts/check_release.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_files():
    return {
        "PKG-INFO": METADATA,
        "LICENSE": LICENSE,
        "README.md": b"example",
        "src/taco_shell_rag/__init__.py": b"",
    }


def make_sdist(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo("example-0.1.0/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_source_archive_matches_its_public_manifest(release, source_files, tmp_path):
    path = make_sdist(tmp_path / "source.tar.gz", source_files)
    assert release.inspect_sdist(path, ["LICENSE", "README.md", "src"], LICENSE) == 4


@pytest.mark.parametrize(
    "name", [".env", "src/.env.private", "HANDOFF.md", "src/../outside", "vendor/tool.py"]
)
def test_private_or_unlisted_source_files_are_rejected(release, source_files, tmp_path, name):
    source_files[name] = b"not a real secret"
    path = make_sdist(tmp_path / "source.tar.gz", source_files)
    with pytest.raises(ValueError):
        release.inspect_sdist(path, ["LICENSE", "README.md", "src"], LICENSE)


def test_missing_source_paths_and_mismatched_license_are_rejected(release, source_files, tmp_path):
    path = make_sdist(tmp_path / "source.tar.gz", source_files)
    with pytest.raises(ValueError, match="missing"):
        release.inspect_sdist(path, ["LICENSE", "README.md", "src", "models.toml"], LICENSE)
    with pytest.raises(ValueError, match="license differs"):
        release.inspect_sdist(path, ["LICENSE", "README.md", "src"], b"another license")


def test_archive_links_are_not_treated_as_public_files(release, tmp_path):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("example-0.1.0/README.md")
        member.type, member.linkname = tarfile.SYMTYPE, "/private/file"
        archive.addfile(member)
    with pytest.raises(ValueError, match="link"):
        release.inspect_sdist(path, ["README.md"], LICENSE)


@pytest.fixture
def wheel_files():
    root = "taco_shell_rag-0.1.0.dist-info"
    return {
        "taco_shell_rag/__init__.py": b"",
        "taco_shell_rag/cli.py": b"",
        f"{root}/METADATA": METADATA,
        f"{root}/WHEEL": b"",
        f"{root}/RECORD": b"",
        f"{root}/entry_points.txt": b"",
        f"{root}/licenses/LICENSE": LICENSE,
    }


def make_wheel(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def test_wheel_contains_only_runtime_and_license_metadata(release, wheel_files, tmp_path):
    path = make_wheel(tmp_path / "example.whl", wheel_files)
    assert release.inspect_wheel(path, LICENSE) == 7


@pytest.mark.parametrize("name", [".env", "taco_shell_rag/.env", "vendor/tool.py", "../outside.py"])
def test_wheel_private_and_nonruntime_files_are_rejected(release, wheel_files, tmp_path, name):
    wheel_files[name] = b"not a real secret"
    path = make_wheel(tmp_path / "example.whl", wheel_files)
    with pytest.raises(ValueError):
        release.inspect_wheel(path, LICENSE)


def test_license_expression_must_match_the_project(release):
    with pytest.raises(ValueError, match="Apache-2.0"):
        release.check_metadata(METADATA.replace(b"Apache-2.0", b"MIT"))


@pytest.mark.parametrize("name", ["", ".", "./src/module.py", "src//module.py", "src\\module.py"])
def test_noncanonical_paths_are_rejected(release, name):
    with pytest.raises(ValueError, match="unsafe path"):
        release.public_path(name)


def test_wheel_symlink_cannot_masquerade_as_runtime_code(release, wheel_files, tmp_path):
    path = make_wheel(tmp_path / "example.whl", wheel_files)
    with zipfile.ZipFile(path, "a") as archive:
        member = zipfile.ZipInfo("taco_shell_rag/link.py")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "/private/file")
    with pytest.raises(ValueError, match="link"):
        release.inspect_wheel(path, LICENSE)


def test_source_release_uses_explicit_public_paths():
    with Path("pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    paths = project["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert paths and all(Path(path).exists() for path in paths)
    assert {"models.toml", "LICENSE", "scripts/check_release.py"} <= set(paths)
    assert ".agents/skills/taco-install" in paths
    assert ".claude/skills/taco-install" in paths
    assert ".agents" not in paths and ".claude" not in paths
    assert "scripts" not in paths and "docs" not in paths
