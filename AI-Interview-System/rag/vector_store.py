"""
A small FAISS vector store over the interview question bank.
Builds once, caches the index + metadata to disk, and supports
domain-filtered similarity search.
"""
import os
import json
import numpy as np
import faiss

from rag.embeddings import embed_document


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []  # list[dict] aligned with index rows

    def build(self, items: list[dict]):
        """items: list of dicts with at least 'question' and 'domain'."""
        vectors = []
        for item in items:
            text = item["question"] + " " + " ".join(item.get("key_points", []))
            vectors.append(embed_document(text))
        arr = np.array(vectors, dtype="float32")
        self.index = faiss.IndexFlatL2(arr.shape[1])
        self.index.add(arr)
        self.metadata = items

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)

    def load(self, dir_path: str) -> bool:
        index_path = os.path.join(dir_path, "index.faiss")
        meta_path = os.path.join(dir_path, "metadata.json")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)
        return True

    def search(self, query_embedding: list[float], top_k: int = 3, domain: str | None = None):
        if self.index is None:
            return []
        # Over-fetch when filtering by domain since FAISS doesn't filter natively.
        k = top_k * 6 if domain else top_k
        k = min(k, len(self.metadata)) or 1
        query = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query, k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            if domain and item.get("domain") != domain:
                continue
            results.append(item)
            if len(results) >= top_k:
                break
        return results


def load_or_build_store(question_bank_path: str, cache_dir: str) -> VectorStore:
    store = VectorStore()
    if store.load(cache_dir):
        return store
    with open(question_bank_path) as f:
        items = json.load(f)
    store.build(items)
    store.save(cache_dir)
    return store
