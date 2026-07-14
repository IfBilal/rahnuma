from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[str], source: str, university: str) -> list[Document]:
    """
    Split page-level text into embedding-sized chunks, one Document per chunk.
    Each chunk keeps track of which page it came from, for citations later.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks: list[Document] = []
    for page_num, page_text in enumerate(pages, start=1):
        if not page_text:
            continue  # skip blank pages (cover pages, section dividers)

        page_chunks = splitter.split_text(page_text)
        for chunk_text in page_chunks:
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source,
                        "university": university,
                        "page": page_num,
                    },
                )
            )

    return chunks


if __name__ == "__main__":
    from ingestion.loader import load_pdf

    pages = load_pdf("corpus/nust/nust_prospectus.pdf")
    chunks = chunk_pages(pages, source="nust_prospectus.pdf", university="NUST")

    print(f"Produced {len(chunks)} chunks from {len(pages)} pages")
    print(f"\nFirst non-trivial chunk:\n{chunks[5]}")
