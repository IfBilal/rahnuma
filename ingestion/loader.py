from pypdf import PdfReader


def load_pdf(path: str) -> list[str]:
    """
    Extract text from a single PDF, one string per page.
    """
    reader = PdfReader(path)
    pages = [page.extract_text() for page in reader.pages]
    avg_chars = sum(len(page) for page in pages if page is not None) / len(pages)
    if avg_chars < 50:
        print(f"Warning: PDF {path} seems to have very little text (avg {avg_chars:.1f} chars/page). It may be image-based and require OCR.")
    return pages

if __name__ == "__main__":
    pages = load_pdf("corpus/nust/nust_prospectus.pdf")
    print(f"Extracted {len(pages)} pages")
    print(f"Page 1 preview:\n{pages[0][:300]}")