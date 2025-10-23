from __future__ import annotations
import time
import httpx
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from rag.utils import (
QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDINGS_MODEL, LANG_FILTER
)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBEDDINGS_MODEL)

SYSTEM_PROMPT = (
"you are a technical assistant. Respond factually, citing your sources"
)

def retrieve(query: str, k: int = 3, lang: str = LANG_FILTER):
    t0 = time.time()
    qvec = embedder.encode([query])[0]
    flt = models.Filter(must=[models.FieldCondition(
    key="lang", match=models.MatchValue(value=lang)
    )]) if lang else None

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec.tolist(),
        limit=k,
        query_filter=flt,
        search_params=models.SearchParams(hnsw_ef=128, exact=False)
    )

    latency_ms = int((time.time() - t0) * 1000)
    docs = [{
        "text": h.payload.get("text"),
        "doc_id": h.payload.get("doc_id"),
        "source": h.payload.get("source"),
        "score": float(h.score),
    } for h in hits]
    return docs, latency_ms


def call_ollama(prompt: str) -> tuple[str, int]:
    t0 = time.time()
    with httpx.Client(timeout=60.0) as client_http:
        r = client_http.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        txt = r.json()["response"]
    return txt, int((time.time() - t0) * 1000)


def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in contexts])
    return (
    f"System: {SYSTEM_PROMPT}\n\n"
    f"Contexte:\n{ctx}\n\n"
    f"Question: {question}\n"
    f"Answer in 5-7 sentences maximum and cite your sources in square brackets."
    )

def answer_question(question: str, lang: str = LANG_FILTER, k: int = 3):
    docs, ret_latency = retrieve(question, k=k, lang=lang)
    prompt = build_prompt(question, docs)
    answer, llm_latency = call_ollama(prompt)
    return {
    "answer": answer,
    "sources": [{"source": d["source"], "doc_id": d["doc_id"], "score": d["score"]} for d in docs],
    "retrieval_latency_ms": ret_latency,
    "llm_latency_ms": llm_latency,
    }