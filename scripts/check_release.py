import argparse
import json
import stat
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath


def public_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or "\\" in name
        or ".." in path.parts
        or str(path) != name
        or set(path.parts) & {".git", ".venv", "__pycache__", "_workspace"}
    ):
        raise ValueError("archive contains an unsafe path")
    filename = path.name.lower()
    if (
        filename == ".env"
        or (filename.startswith(".env.") and filename != ".env.example")
        or filename.startswith("handoff")
        or filename in {"agents.local.md", "current-state.md", "session-log.md"}
        or filename.endswith((".local.toml", ".pyc", ".pyo"))
    ):
        raise ValueError("archive contains a private or generated file")
    return path


def check_metadata(content: bytes):
    metadata = BytesParser(policy=default).parsebytes(content)
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError("package metadata must declare Apache-2.0")
    if metadata.get_all("License-File", []) != ["LICENSE"]:
        raise ValueError("package metadata must include LICENSE")


def inspect_sdist(path: Path, public_paths: list[str], license_text: bytes) -> int:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        paths = [public_path(member.name.rstrip("/")) for member in members]
        roots = {name.parts[0] for name in paths}
        if len(roots) != 1:
            raise ValueError("source archive must have one root directory")
        files = {}
        for member, name in zip(members, paths, strict=True):
            if member.isdir():
                continue
            # 배포본을 풀지 않고 검사한다. 링크로 포함 경로 검사를 우회할 수 없게 한다.
            # Inspect without extraction; links must not bypass the public-path boundary.
            if not member.isfile() or len(name.parts) < 2:
                raise ValueError("source archive contains a link or non-file entry")
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            public_path(relative)
            if relative in files:
                raise ValueError("source archive contains duplicate paths")
            if relative != "PKG-INFO" and not any(
                relative == allowed or relative.startswith(allowed + "/")
                for allowed in public_paths
            ):
                raise ValueError("source archive contains a path outside the public manifest")
            files[relative] = member
        for allowed in [*public_paths, "PKG-INFO", "LICENSE"]:
            if not any(name == allowed or name.startswith(allowed + "/") for name in files):
                raise ValueError("source archive is missing a required public path")
        check_metadata(archive.extractfile(files["PKG-INFO"]).read())
        if archive.extractfile(files["LICENSE"]).read() != license_text:
            raise ValueError("source archive license differs from the project license")
    return len(files)


def inspect_wheel(path: Path, license_text: bytes) -> int:
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in items):
            raise ValueError("wheel contains a link")
        names = [public_path(item.filename) for item in items if not item.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate paths")
        metadata_roots = {name.parts[0] for name in names if name.parts[0].endswith(".dist-info")}
        if len(metadata_roots) != 1:
            raise ValueError("wheel must have one metadata directory")
        metadata_root = metadata_roots.pop()
        metadata_files = {
            f"{metadata_root}/{name}"
            for name in ["METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE"]
        }
        for name in names:
            runtime = name.parts[0] == "taco_shell_rag" and name.suffix == ".py"
            if not runtime and str(name) not in metadata_files:
                raise ValueError("wheel contains a file outside the runtime package")
        required = metadata_files | {"taco_shell_rag/__init__.py", "taco_shell_rag/cli.py"}
        if not required <= {str(name) for name in names}:
            raise ValueError("wheel is missing runtime or metadata files")
        check_metadata(archive.read(f"{metadata_root}/METADATA"))
        if archive.read(f"{metadata_root}/licenses/LICENSE") != license_text:
            raise ValueError("wheel license differs from the project license")
    return len(names)


def main():
    parser = argparse.ArgumentParser(description="Check built source and wheel contents")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    public_paths = project["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    source, wheels = list(args.directory.glob("*.tar.gz")), list(args.directory.glob("*.whl"))
    if len(source) != 1 or len(wheels) != 1:
        parser.exit(1, "release: expected exactly one source archive and one wheel\n")
    try:
        license_text = (root / "LICENSE").read_bytes()
        result = {
            "source_files": inspect_sdist(source[0], public_paths, license_text),
            "wheel_files": inspect_wheel(wheels[0], license_text),
        }
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f"release: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
