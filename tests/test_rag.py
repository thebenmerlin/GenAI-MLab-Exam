import pytest

import rag
from rag import LocalTfidfIndex, RAGError, VectorIndex, _split_long_text, build_document_index, extract_chunks, retrieve


def test_split_text_creates_overlapping_chunks():
    text = " ".join(["This is a complete research sentence."] * 120)
    chunks = _split_long_text(text, chunk_size=220, overlap=40)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert any(chunks[0][-20:] in next_chunk for next_chunk in chunks[1:])


def test_text_file_preserves_a_document_location():
    chunks = extract_chunks(b"This is enough readable text for a tiny document." * 3, "paper.txt")

    assert chunks
    assert all(chunk.location == "Text document" for chunk in chunks)


def test_rejects_unsupported_document_type():
    with pytest.raises(RAGError, match="PDF, DOCX, or TXT"):
        extract_chunks(b"not a document", "paper.csv")


def test_vector_index_returns_best_match():
    index = VectorIndex([[1, 0], [0, 1], [0.8, 0.2]])
    matches = index.search([0.95, 0.05], limit=2)

    assert matches[0][0] == 0
    assert matches[0][1] > matches[1][1]


def test_local_tfidf_fallback_returns_matching_chunk():
    index = LocalTfidfIndex(
        [
            "The retrieval system reports 84 percent accuracy.",
            "The paper discusses a neural network training schedule.",
        ]
    )

    matches = index.search("What accuracy does the retrieval system report?", limit=1)

    assert matches[0][0] == 0
    assert matches[0][1] > 0


def test_embedding_quota_uses_local_retrieval_fallback(monkeypatch):
    def quota_error(*_args, **_kwargs):
        raise RAGError("Gemini's free-tier limit was reached. Wait briefly and try again.")

    monkeypatch.setattr(rag, "_embed_many", quota_error)
    index = build_document_index(
        b"The retrieval system achieved 84 percent accuracy." * 3,
        "paper.txt",
        "test-key",
    )

    matches = retrieve(index, "What accuracy did the retrieval system achieve?", "test-key")

    assert index.retrieval_mode == "local_fallback"
    assert matches
