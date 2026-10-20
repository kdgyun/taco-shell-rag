import httpx
import pytest

from taco_shell_rag.indexer.sources import FolderSource, RestSource


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


def test_rest_cursor_and_explicit_field_mapping():
    requests = []

    def respond(request):
        requests.append(request)
        cursor = request.url.params.get("cursor")
        return httpx.Response(
            200,
            json={
                "items": [{"key": "second" if cursor else "first", "body": "exact text"}],
                "next_cursor": None if cursor else "page-two",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    source = RestSource(
        "api", "https://example.com/docs", {"id": "key", "text": "body"}, client=client
    )
    result = source.scan(None)
    assert [d.id for d in result.documents] == ["first", "second"]
    assert requests[1].url.host == "example.com"
    assert requests[1].url.params["cursor"] == "page-two"


@pytest.mark.parametrize(
    "response",
    [
        {"items": [], "next_cursor": "same"},
        {"items": [], "next_cursor": "https://another-host/"},
    ],
)
def test_rest_does_not_follow_urls_or_repeat_cursors(response):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    source = RestSource(
        "api",
        "https://example.com/docs",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    with pytest.raises(ValueError, match="cursor"):
        source.scan(None)
    assert all(r.url.host == "example.com" for r in requests)


def test_rest_does_not_follow_redirect_or_send_credentials_to_redirect(monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "private-value")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(302, headers={"location": "https://attacker.example"})
        )
    )
    source = RestSource("api", "https://example.com/docs", token_env="DEMO_TOKEN", client=client)
    with pytest.raises(ValueError, match="302"):
        source.scan(None)


def test_rest_rejects_embedded_credentials_and_nonstring_fields():
    with pytest.raises(ValueError, match="credentials"):
        RestSource("api", "https://user:pass@example.com/docs")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"items": [{"id": 1, "text": "body"}]})
        )
    )
    with pytest.raises(ValueError, match="strings"):
        RestSource("api", "https://example.com/docs", client=client).scan(None)
