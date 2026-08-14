"""
retriever.py — Semantic search over the ChromaDB policy collection.

Embedding model: all-MiniLM-L6-v2 (via ChromaDB ONNX backend)
Uses the same DefaultEmbeddingFunction as ingest.py so embeddings are consistent.
Exposes a single public function: retrieve(query, top_k).
"""

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from constants import DB_DIR, COLLECTION_NAME, TOP_K

# ── Module-level singleton ────────────────────────────────────────────────────

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        # Must use the same embedding function used during ingestion
        ef = DefaultEmbeddingFunction()
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    return _collection


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top_k most similar document chunks.

    ChromaDB embeds the query text internally using DefaultEmbeddingFunction
    (all-MiniLM-L6-v2 via ONNX Runtime — same model as sentence-transformers).

    Returns a list of dicts, each with:
        id:       str   — the chunk ID (e.g. "doc_01_chunk_0")
        document: str   — the chunk text
        source:   str   — the source filename
        distance: float — cosine distance (lower = more similar)
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "id":       chunk_id,
            "document": doc,
            "source":   meta["source"],
            "distance": round(float(dist), 4),
        })

    return chunks
