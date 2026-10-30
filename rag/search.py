from __future__ import annotations
import time
from functools import lru_cache
import httpx
from qdrant_client import QdrantClient, models
from rag.utils import (
QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, EMBEDDINGS_MODEL, LANG_FILTER
)

@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@lru_cache(maxsize=1)
def get_embedder():
    # Le modèle est chargé à la première recherche, pas au démarrage de l'interface.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDINGS_MODEL)

SYSTEM_PROMPT = (
"you are a technical assistant. Respond factually, citing your sources"
)


def correspondence_percent(score: float) -> float:
    """Cosinus positif en pourcentage ; ce n'est pas une confiance calibrée."""
    return round(max(0.0, min(1.0, score)) * 100, 1)

def retrieve(query: str, k: int = 3, lang: str = LANG_FILTER):
    t0 = time.time()
    qvec = get_embedder().encode([query])[0]
    flt = models.Filter(must=[models.FieldCondition(
    key="lang", match=models.MatchValue(value=lang)
    )]) if lang else None

    hits = get_client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=qvec.tolist(),
        limit=k,
        query_filter=flt,
        search_params=models.SearchParams(hnsw_ef=128, exact=False)
    ).points

    latency_ms = int((time.time() - t0) * 1000)
    docs = [{
        "text": h.payload.get("text"),
        "doc_id": h.payload.get("doc_id"),
        "source": h.payload.get("source"),
        "page": h.payload.get("page"),
        "score": float(h.score),
        "correspondence_pct": correspondence_percent(float(h.score)),
    } for h in hits]
    return docs, latency_ms


def call_ollama(prompt: str) -> tuple[str, int]:
    t0 = time.time()
    # Le chargement initial et la génération sur CPU peuvent dépasser une minute.
    timeout = httpx.Timeout(OLLAMA_TIMEOUT_SECONDS, connect=5.0)
    with httpx.Client(timeout=timeout) as client_http:
        r = client_http.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        txt = r.json()["response"]
    return txt, int((time.time() - t0) * 1000)


def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[Source: {c['source']}" + (f", page {c['page']}" if c.get("page") else "")
        + f"]\n{c['text']}" for c in contexts
    )
    return (
    f"System: {SYSTEM_PROMPT}\n\n"
    f"Contexte:\n{ctx}\n\n"
    f"Question: {question}\n"
    f"Answer in 5-7 sentences maximum and cite your sources in square brackets."
    )

def answer_question(question: str, lang: str = LANG_FILTER, k: int = 3):
    docs, ret_latency = retrieve(question, k=k, lang=lang)
    if not docs:
        return {
            "answer": "Aucun extrait trouvé pour cette langue. Vérifiez le filtre et l’indexation du corpus.",
            "sources": [],
            "retrieval_latency_ms": ret_latency,
            "llm_latency_ms": 0,
        }
    prompt = build_prompt(question, docs)
    answer, llm_latency = call_ollama(prompt)
    return {
    "answer": answer,
    "sources": docs,
    "retrieval_latency_ms": ret_latency,
    "llm_latency_ms": llm_latency,
    }
