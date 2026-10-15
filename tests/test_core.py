from dataclasses import replace

import pytest

from taco_shell_rag.core.chunking import split_document
from taco_shell_rag.core.contracts import Document, EmbeddingSpaceMismatch, Snapshot


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
    calls = kb.embedder.embedded
    assert kb.sync()[source.id]["changed"] == 0
    assert kb.embedder.embedded == calls
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
    kb.embedder.fail = True
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
    kb.embedder.cpu()
    with pytest.raises(ValueError, match="CUDA"):
        kb.sync()
    kb.embedder.info = replace(kb.embedder.info, device="cuda:0")
    kb.sync()
    kb.embedder.info = replace(kb.embedder.info, revision="different")
    with pytest.raises(ValueError, match="embedding space"):
        kb.search("refund")


def test_writer_lock_and_duplicate_inventory(kb):
    source = attach(kb, ChangingSource())
    with kb.ledger.writer(), pytest.raises(RuntimeError, match="writer lock"):
        kb.sync()
    source.documents *= 2
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.status()["documents"] == 0


def test_reranker_choice_is_independent_of_indexing(kb):
    attach(kb, ChangingSource())
    kb.reranker.fail = True
    assert kb.sync()["notes"]["status"] == "ok"
    assert kb.reranker.calls == 0
    with pytest.raises(ConnectionError):
        kb.search("refund")
    calls = kb.embedder.embedded
    kb.reranker.fail = False
    kb.reranker.info = replace(
        kb.reranker.info, model="another-series", revision="another-revision"
    )
    assert kb.sync()["notes"]["changed"] == 0
    assert kb.embedder.embedded == calls
    assert kb.search("refund")["reranker"]["model"] == "another-series"


@pytest.mark.parametrize(
    "encoding",
    [
        {"query_prompt": "query: "},
        {"document_prompt": "passage: "},
        {"max_length": 256},
        {"precision": "float16"},
    ],
)
def test_changed_embedding_options_cannot_reuse_old_vectors(kb, encoding):
    attach(kb, ChangingSource())
    kb.sync()
    kb.embedder.info = replace(kb.embedder.info, encoding=encoding)
    with pytest.raises(ValueError, match="embedding space"):
        kb.search("refund")
    with pytest.raises(ValueError, match="embedding space"):
        kb.sync()


def test_disabled_reranker_does_not_call_it_or_claim_rerank_scores(kb):
    attach(kb, ChangingSource())
    kb.rerank_enabled = False
    kb.reranker.fail = True
    kb.sync()
    result = kb.search("refund")
    assert result["score_kind"] == "cosine"
    assert result["reranker"] is None
    assert kb.reranker.calls == 0


def test_reranker_cpu_response_is_not_accepted(kb):
    attach(kb, ChangingSource())
    kb.sync()
    kb.reranker.info = replace(kb.reranker.info, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        kb.search("refund")


@pytest.mark.parametrize(
    "changes",
    [{"revision": "another-revision"}, {"encoding": {"precision": "float16"}}, {"device": "cpu"}],
)
def test_execution_space_mismatch_never_reaches_vector_search(kb, monkeypatch, changes):
    attach(kb, ChangingSource())
    kb.sync()
    original = kb.embedder.embed

    def switched(texts, purpose):
        result = original(texts, purpose)
        return replace(result, space=replace(result.space, **changes))

    def unexpected(*_):
        raise AssertionError("incompatible vectors must not reach the index")

    monkeypatch.setattr(kb.embedder, "embed", switched)
    monkeypatch.setattr(kb.index, "search", unexpected)
    with pytest.raises(ValueError):
        kb.search("refund")


@pytest.mark.parametrize(
    "changes",
    [{"revision": "another-revision"}, {"encoding": {"max_length": 256}}, {"device": "cpu"}],
)
def test_model_change_between_batches_preserves_publication_and_checkpoint(
    kb, monkeypatch, changes
):
    source = attach(kb, ChangingSource())
    kb.sync()
    published = kb.ledger.get(source.documents[0].key)
    points = kb.index.client.count("test").count
    source.documents = [replace(source.documents[0], text="updated refund policy " * 100)]
    source.checkpoint = "second"
    original = kb.embedder.embed
    calls = 0

    def switched(texts, purpose):
        nonlocal calls
        calls += 1
        result = original(texts, purpose)
        return replace(result, space=replace(result.space, **changes)) if calls == 2 else result

    monkeypatch.setattr(kb.embedder, "embed", switched)
    assert kb.sync()[source.id]["status"] == "error"
    assert calls == 2
    assert kb.ledger.checkpoint(source.id) == "first"
    assert kb.ledger.get(source.documents[0].key) == published
    assert kb.index.client.count("test").count == points


def test_same_dimension_does_not_make_execution_spaces_compatible(kb, monkeypatch):
    attach(kb, ChangingSource())
    kb.sync()
    original = kb.embedder.embed

    def switched(texts, purpose):
        result = original(texts, purpose)
        return replace(result, space=replace(result.space, model="another-model"))

    monkeypatch.setattr(kb.embedder, "embed", switched)
    with pytest.raises(EmbeddingSpaceMismatch, match="execution space changed"):
        kb.search("refund")
