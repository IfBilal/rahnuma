from langchain_core.documents import Document

from ingestion.chunker import chunk_pages


def test_chunk_pages_keeps_citation_metadata_and_skips_blank_pages():
    pages = ["", "Admission merit criteria " * 100]

    chunks = chunk_pages(pages, source="prospectus.pdf", university="NUST")

    assert chunks
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.metadata["source"] == "prospectus.pdf" for chunk in chunks)
    assert all(chunk.metadata["university"] == "NUST" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)


def test_chunk_pages_creates_overlapping_embedding_sized_chunks():
    chunks = chunk_pages(["a" * 2_100], source="x.pdf", university="FAST")

    assert len(chunks) == 3
    assert all(len(chunk.page_content) <= 1_000 for chunk in chunks)
