from dataclasses import replace

import pytest

from taco_shell_rag.core.contracts import Document, Snapshot


class Notes:
    id = "notes"

    def __init__(self):
        self.documents = [Document("a", self.id, "old policy")]

    def scan(self, checkpoint):
        return Snapshot(self.documents, "snapshot")


def test_partial_vector_write_never_publishes_and_orphans_are_cleaned(kb, monkeypatch):
    source = Notes()
    kb.sources = {source.id: source}
    original = kb.index.upsert

    def partial(chunks, vectors):
        original(chunks[:1], vectors[:1])
        raise RuntimeError("failed after one point")

    monkeypatch.setattr(kb.index, "upsert", partial)
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.status()["documents"] == 0
    assert kb.search("policy")["results"] == []
    assert kb.ledger.status()["pending_cleanup"] == 1
    source.documents = [replace(source.documents[0], text="new version")]
    monkeypatch.setattr(kb.index, "upsert", original)
    assert kb.sync()[source.id]["status"] == "ok"
    assert kb.index.client.count("test").count == 1
    assert kb.ledger.status()["pending_cleanup"] == 0


def test_invalid_embedding_values_do_not_replace_published_document(kb, monkeypatch):
    source = Notes()
    kb.sources = {source.id: source}
    kb.sync()
    old = kb.ledger.get(source.documents[0].key)
    source.documents = [replace(source.documents[0], text="different policy")]
    monkeypatch.setattr(kb.embedder, "embed", lambda *_: [[float("nan"), 0.0, 0.0]])
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.get(source.documents[0].key) == old


def test_new_chunk_settings_reindex_even_when_source_is_unchanged(kb):
    source = Notes()
    kb.sources = {source.id: source}
    kb.sync()
    old = kb.ledger.get(source.documents[0].key)[1][0].version
    kb.chunk_size = 80
    assert kb.sync()[source.id]["changed"] == 1
    assert kb.ledger.get(source.documents[0].key)[1][0].version != old


def test_source_filter_and_document_scope_cannot_read_retired_source(kb):
    source = Notes()
    kb.sources = {source.id: source}
    kb.sync()
    chunk = kb.search("policy")["results"][0]["chunk"]
    kb.sources = {}
    assert kb.search("policy")["results"] == []
    with pytest.raises(ValueError, match="unavailable"):
        kb.get_document(chunk["document_key"], chunk["version"])
