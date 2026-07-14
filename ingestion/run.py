"""
Ingestion runner: walks corpus/, and for every PDF, loads -> chunks -> stores it.

corpus/ layout is corpus/<university>/<file>.pdf, e.g. corpus/fast/admission_information.pdf
so the parent folder name doubles as the university label.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ingestion.chunker import chunk_pages
from ingestion.loader import load_pdf
from ingestion.store import get_vector_store

CORPUS_DIR = Path("corpus")


def ingest_all():
    store = get_vector_store()
    for university_dir in CORPUS_DIR.iterdir():
        if not university_dir.is_dir():
            continue
        university_name=university_dir.name
        for pdf in university_dir.glob("*.pdf"):
            try:
                pages = load_pdf(pdf)
                chunks = chunk_pages(pages, source=pdf.name, university=university_name)
                store.add_documents(chunks)
                print(f"{university_name}: {pdf.name} -> {len(chunks)} chunks")
            except Exception as e:
                print(f"FAILED: {university_name}: {pdf.name} -> {e}")
                continue


if __name__ == "__main__":
    ingest_all()
