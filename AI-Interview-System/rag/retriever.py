"""Turns the candidate's resume skills into a retrieval query so question
generation is grounded in the question bank instead of pure LLM imagination."""
from rag.embeddings import embed_query
from rag.vector_store import VectorStore


def retrieve_relevant_questions(store: VectorStore, skills: list[str], domain: str, top_k: int = 3):
    query_text = (
        f"Candidate skills: {', '.join(skills) if skills else 'general candidate'}. "
        f"Find relevant {domain} interview questions for these skills."
    )
    query_embedding = embed_query(query_text)
    return store.search(query_embedding, top_k=top_k, domain=domain)
