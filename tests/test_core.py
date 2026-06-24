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


def test_writer_lock_and_duplicate_inventory(kb):
    source = attach(kb, ChangingSource())
    with kb.ledger.writer(), pytest.raises(RuntimeError, match="writer lock"):
        kb.sync()
    source.documents *= 2
    assert kb.sync()[source.id]["status"] == "error"
    assert kb.ledger.status()["documents"] == 0
