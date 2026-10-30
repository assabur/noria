from rag.search import retrieve


QUERIES = [
("what is an airflow Dag?", "Airflow")
]


def test_retrieval_quality():
    hits = 0
    for q, keyword in QUERIES:
        docs, _ = retrieve(q, k=3, lang="fr")
        if any(keyword.lower() in (d["text"] or "").lower() for d in docs):
            hits += 1
    assert hits >= 1, ""
