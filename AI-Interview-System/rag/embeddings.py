"""Thin wrapper kept separate so the RAG layer has a clear single entry point
for turning text into vectors, independent of which LLM provider is used."""
from utils import embed_text


def embed_document(text: str):
    return embed_text(text, task_type="retrieval_document")


def embed_query(text: str):
    return embed_text(text, task_type="retrieval_query")
