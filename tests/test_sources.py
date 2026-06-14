import pytest

from taco_shell_rag.adapters.sources import FolderSource


def test_folder_ignores_hidden_secret_files_and_symlinks(tmp_path):
    (tmp_path / "guide.md").write_text("public notes")
    (tmp_path / ".env").write_text("SECRET=value")
    hidden = tmp_path / ".private"
    hidden.mkdir()
    (hidden / "secret.md").write_text("hidden")
    (tmp_path / "linked.md").symlink_to(hidden / "secret.md")
    result = FolderSource("docs", tmp_path).scan(None)
    assert [d.id for d in result.documents] == ["guide.md"]
    assert result.documents[0].text == "public notes"


def test_folder_oversize_is_failure_not_complete_empty_inventory(tmp_path):
    (tmp_path / "guide.md").write_text("too much text")
    with pytest.raises(ValueError, match="bound"):
        FolderSource("docs", tmp_path, max_bytes=4).scan(None)
