from dataclasses import replace

from taco_shell_rag.core.contracts import Document, Snapshot
from taco_shell_rag.indexer.catalog import Catalog


class Notes:
    id = "notes"

    def __init__(self):
        self.documents = [Document("policy", self.id, "original policy")]
        self.complete = True

    def scan(self, checkpoint):
        return Snapshot(self.documents, "snapshot", complete=self.complete)


def text_rows(catalog):
    with catalog.connect() as db:
        return [
            tuple(row)
            for row in db.execute(
                "SELECT document_key,version,text FROM chunk_fts ORDER BY document_key,chunk_index"
            )
        ]


def test_keyword_index_migrates_existing_publications_without_reembedding(kb):
    notes = Notes()
    kb.sources = {notes.id: notes}
    kb.sync()
    before = text_rows(kb.ledger)
    embedded = kb.embedder.embedded
    with kb.ledger.connect() as db:
        db.execute("DROP TABLE chunk_fts")
        db.execute("DELETE FROM settings WHERE key='lexical_schema'")
    migrated = Catalog(kb.ledger.path)
    assert text_rows(migrated) == before
    assert kb.embedder.embedded == embedded
    assert Catalog(kb.ledger.path).status() == migrated.status()


def test_lexical_failure_rolls_back_document_activation(kb, monkeypatch):
    notes = Notes()
    kb.sources = {notes.id: notes}
    kb.sync()
    before = kb.ledger.get(notes.documents[0].key)
    lexical = text_rows(kb.ledger)
    notes.documents = [replace(notes.documents[0], text="replacement policy")]

    def fail(*_):
        raise RuntimeError("simulated FTS failure")

    monkeypatch.setattr(kb.ledger, "_index_text", fail)
    assert kb.sync()[notes.id]["status"] == "error"
    assert kb.ledger.get(notes.documents[0].key) == before
    assert text_rows(kb.ledger) == lexical


def test_incomplete_inventory_keeps_lexical_evidence_and_complete_delete_removes_it(kb):
    notes = Notes()
    kb.sources = {notes.id: notes}
    kb.sync()
    before = text_rows(kb.ledger)
    notes.documents = []
    notes.complete = False
    assert kb.sync()[notes.id]["status"] == "error"
    assert text_rows(kb.ledger) == before
    notes.complete = True
    assert kb.sync()[notes.id]["status"] == "ok"
    assert text_rows(kb.ledger) == []


def test_partial_vector_write_cannot_publish_lexical_content(kb, monkeypatch):
    notes = Notes()
    kb.sources = {notes.id: notes}
    original = kb.index.upsert

    def partial(chunks, vectors):
        original(chunks, vectors)
        raise RuntimeError("write acknowledgment lost")

    monkeypatch.setattr(kb.index, "upsert", partial)
    assert kb.sync()[notes.id]["status"] == "error"
    assert text_rows(kb.ledger) == []
    assert kb.ledger.status()["documents"] == 0
