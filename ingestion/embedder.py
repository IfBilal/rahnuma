from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

# Persistent cache (not /tmp) so the model is downloaded once, ever —
# not re-fetched every time /tmp gets cleared on reboot.
CACHE_DIR = ".fastembed_cache"


class LocalEmbeddings(Embeddings):
    """
    Local embedding model via fastembed (ONNX Runtime, no torch).
    Runs entirely on-device after the one-time model download — no network
    calls per embedding, no rate limits, no external quota.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name, cache_dir=CACHE_DIR)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        # fastembed's ONNX Runtime backend accumulates memory across calls and
        # never releases it — embedding a whole file (hundreds of chunks) in
        # one call is what caused the ingestion run to eventually run out of
        # RAM. Small fixed batches keep peak memory bounded and predictable.
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(vec.tolist() for vec in self.model.embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


if __name__ == "__main__":
    embedder = LocalEmbeddings()
    vectors = embedder.embed_documents(["hello world", "FAST merit formula"])
    print(f"Embedded {len(vectors)} texts, each of length {len(vectors[0])}")
