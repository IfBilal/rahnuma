import os

from langchain_core.documents import Document
from langchain_postgres import PGVector

from ingestion.embedder import LocalEmbeddings

COLLECTION_NAME = "rahnuma_corpus"


def get_vector_store() -> PGVector:
    """
    Returns a PGVector store bound to our Postgres instance.
    add_documents() on this object embeds + inserts in one call —
    no manual embedding step needed, PGVector calls LocalEmbeddings for us.
    """
    embeddings = LocalEmbeddings()
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    from ingestion.chunker import chunk_pages
    from ingestion.loader import load_pdf

    pages = load_pdf("corpus/nust/nust_prospectus.pdf")
    chunks = chunk_pages(pages, source="nust_prospectus.pdf", university="NUST")

    store = get_vector_store()
    ids = store.add_documents(chunks[:10])  # small batch first, sanity check
    print(f"Inserted {len(ids)} chunks into pgvector")

    results = store.similarity_search("What is the merit criteria?", k=3)
    for doc in results:
        print(f"\n--- page {doc.metadata['page']} ---")
        print(doc.page_content[:200])
