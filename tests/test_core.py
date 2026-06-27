from dataclasses import replace

import pytest

from taco_shell_rag.core.chunking import split_document
from taco_shell_rag.core.contracts import Document, Snapshot


class ChangingSource:
    id = "notes"

    def __init__(self, text="refund policy " * 20):
        self.documents = [Document("a", self.id, text)]
        self.complete = True
        self.checkpoint = "first"

    def scan(self, checkpoint):
        return Snapshot(self.documents, self.checkpoint, self.complete)


def attach(kb, source):
    kb.sources = {source.id: source}
    return source


def test_unicode_chunks_reconstruct_and_namespace_identity():
    document = Document("same", "one", "가나다🙂" * 100)
    chunks = split_document(document, 64, 8)
    assert chunks[0].text + "".join(c.text[8:] for c in chunks[1:]) == document.text
    assert document.key != replace(document, source_id="two").key
    assert chunks[0].version != split_document(document, 65, 8)[0].version


def test_sync_search_read_and_unchanged_skip(kb):
    source = attach(kb, ChangingSource())
    assert kb.sync()[source.id]["changed"] == 1
    calls = kb.inference.embedded
    assert kb.sync()[source.id]["changed"] == 0
    assert kb.inference.embedded == calls
    result = kb.search("refund")
    assert result["results"][0]["chunk"]["text"]
    assert result["score_kind"] == "reranker_raw_uncalibrated"
    assert "text" not in result["results"][0]["document"]
    chunk = result["results"][0]["chunk"]
    read = kb.get_document(chunk["document_key"], chunk["version"], limit=1)
    assert read["next_start"] == 1
    assert len(read["chunks"]) == 1


def test_failed_embedding_keeps_old_publication_and_checkpoint(kb):
    source = attach(kb, ChangingSource())
    kb.sync()
    source.documents = [replace(source.documents[0], text="new text")]
    source.checkpoint = "second"
    kb.inference.fail = True
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.checkpoint(source.id) == "first"
    assert "refund" in kb.ledger.get(source.documents[0].key)[0].text
    assert kb.ledger.status()["sources"][0]["error"] == "RuntimeError"


def test_incomplete_inventory_never_deletes(kb):
    source = attach(kb, ChangingSource())
    kb.sync()
    source.documents, source.complete = [], False
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.status()["documents"] == 1
    source.complete = True
    assert kb.sync()[source.id]["status"] == "ok"
    assert kb.ledger.status()["documents"] == 0
    assert kb.search("refund")["results"] == []


def test_changed_document_replaces_tail_and_old_version_read_fails(kb):
    source = attach(kb, ChangingSource())
    kb.sync()
    old = kb.search("refund")["results"][0]["chunk"]
    source.documents = [replace(source.documents[0], text="new short refund rule")]
    kb.sync()
    assert kb.index.client.count("test").count == 1
    with pytest.raises(ValueError, match="unavailable"):
        kb.get_document(old["document_key"], old["version"])


def test_failed_cleanup_stays_hidden_and_retries(kb, monkeypatch):
    source = attach(kb, ChangingSource())
    kb.sync()
    source.documents = []
    original = kb.index.delete_version
    monkeypatch.setattr(
        kb.index, "delete_version", lambda *_: (_ for _ in ()).throw(RuntimeError())
    )
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.status()["pending_cleanup"] == 1
    assert kb.search("refund")["results"] == []
    monkeypatch.setattr(kb.index, "delete_version", original)
    assert kb.sync()[source.id]["status"] == "ok"
    assert kb.ledger.status()["pending_cleanup"] == 0
    assert kb.index.client.count("test").count == 0


def test_no_cpu_fallback_or_incompatible_model_reuse(kb):
    attach(kb, ChangingSource())
    kb.inference.cpu()
    with pytest.raises(ValueError, match="CUDA"):
        kb.sync()
    kb.inference.info = replace(kb.inference.info, device="cuda:0")
    kb.sync()
    kb.inference.info = replace(kb.inference.info, revision="different")
    with pytest.raises(ValueError, match="embedding space"):
        kb.search("refund")


def test_writer_lock_and_duplicate_inventory(kb):
    source = attach(kb, ChangingSource())
    with kb.ledger.writer(), pytest.raises(RuntimeError, match="writer lock"):
        kb.sync()
    source.documents *= 2
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.status()["documents"] == 0
