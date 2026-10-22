
"""
MILESTONE 2 - Searchable Knowledge Base (ChromaDB)
---------------------------------------------------
Store document chunks as embeddings and search by meaning.
Supports searching all documents or one selected document.
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "safety_docs"

_embedder = None


def embed(texts):
    """Convert text into embedding vectors. Load the model only once."""
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedder.encode(texts).tolist()


def get_collection():
    """Get or create the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def add_document(chunks, source):
    """
    Add a document's chunks to the knowledge base.
    If the same filename already exists, replace its previous chunks.
    """
    col = get_collection()
    source = os.path.basename(source)

    if not chunks:
        return 0

    # Delete existing chunks for this filename before re-uploading.
    old = col.get(where={"source": source}, include=["metadatas"])
    old_ids = old.get("ids", [])

    if old_ids:
        col.delete(ids=old_ids)

    ids = [f"{source}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source, "chunk": i}
        for i in range(len(chunks))
    ]

    col.add(
        ids=ids,
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=metadatas,
    )

    return len(chunks)


def search(question, top_k=5, source=None):
    """
    Find the most relevant chunks.

    Args:
        question: User's question.
        top_k: Maximum number of chunks to return.
        source: Optional filename to restrict the search to one document.
    """
    col = get_collection()

    if col.count() == 0 or not question.strip():
        return []

    top_k = max(1, int(top_k))
    source = os.path.basename(source) if source else None

    # Apply document filtering before retrieval, not afterwards.
    where_filter = {"source": source} if source else None

    # Count records within the selected document when filtering.
    if source:
        matching = col.get(
            where=where_filter,
            include=["metadatas"],
        )
        available_count = len(matching.get("ids", []))
    else:
        available_count = col.count()

    if available_count == 0:
        return []

    candidate_count = min(max(top_k * 3, top_k), available_count)

    query_args = {
        "query_embeddings": embed([question]),
        "n_results": candidate_count,
        "include": ["documents", "metadatas", "distances"],
    }

    if where_filter:
        query_args["where"] = where_filter

    res = col.query(**query_args)

    documents = res.get("documents") or [[]]
    metadatas = res.get("metadatas") or [[]]
    distances = res.get("distances") or [[]]

    if not documents or not documents[0]:
        return []

    results = []

    for text, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        metadata = metadata or {}

        results.append({
            "text": text or "",
            "source": metadata.get("source", "unknown"),
            "chunk": metadata.get("chunk", 0),
            "distance": round(float(distance), 3),
        })

    # Most similar results first. Chunk number breaks ties.
    results.sort(
        key=lambda item: (
            item["distance"],
            item["source"],
            item["chunk"],
        )
    )

    return results[:top_k]


def list_documents():
    """Return a dictionary containing each filename and its chunk count."""
    counts = {}

    metadata_result = get_collection().get(include=["metadatas"])

    for metadata in metadata_result.get("metadatas") or []:
        if not metadata:
            continue

        source = metadata.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1

    return dict(sorted(counts.items()))


def total_chunks():
    """Return the total number of stored chunks."""
    return get_collection().count()


if __name__ == "__main__":
    from documents import process_file

    sample_path = os.path.join(
        BASE_DIR, "data", "sample_safety_manual.txt"
    )

    if os.path.exists(sample_path):
        chunks = process_file(sample_path)
        add_document(chunks, os.path.basename(sample_path))

        for result in search("Do I need a helmet?", top_k=3):
            print(
                f"Distance: {result['distance']} | "
                f"{result['source']} #{result['chunk']}"
            )
            print(result["text"])
            print()
    else:
        print(f"Sample document not found: {sample_path}")
