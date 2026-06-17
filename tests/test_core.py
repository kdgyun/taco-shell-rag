from dataclasses import replace

from taco_shell_rag.core.chunking import split_document
from taco_shell_rag.core.contracts import Document


def test_unicode_chunks_reconstruct_and_namespace_identity():
    document = Document("same", "one", "가나다🙂" * 100)
    chunks = split_document(document, 64, 8)
    assert chunks[0].text + "".join(c.text[8:] for c in chunks[1:]) == document.text
    assert document.key != replace(document, source_id="two").key
    assert chunks[0].version != split_document(document, 65, 8)[0].version
