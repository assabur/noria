from __future__ import annotations
import uuid

from qdrant_client import models
from rag.search import get_client, get_embedder
from rag.utils import QDRANT_COLLECTION


def index_documents(path="data/chunks.parquet", progress=lambda value, stage: None):
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError("Le corpus ne contient aucun extrait.")
    client = get_client()
    vectors = get_embedder().encode(frame["text"].tolist(), batch_size=64, show_progress_bar=True)
    progress(70, "Écriture des vecteurs dans Qdrant")
    # Calculer les vecteurs avant de remplacer l'index existant.
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )
    rows = frame.to_dict("records")
    for row in rows:
        # Les anciens Parquet n'ont pas de page ; les textes ont une valeur nulle.
        page = row.get("page")
        row["page"] = None if pd.isna(page) else int(page)
    for start in range(0, len(rows), 128):
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{row['doc_id']}#{row['chunk_index']}")),
            vector=vector.tolist(), payload=row,
        ) for row, vector in zip(rows[start:start + 128], vectors[start:start + 128])]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        progress(70 + int(25 * min(start + 128, len(rows)) / len(rows)), "Écriture des vecteurs dans Qdrant")
    for field in ("lang", "doc_id"):
        client.create_payload_index(QDRANT_COLLECTION, field_name=field,
                                    field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
    return len(rows)


if __name__ == "__main__":
    count = index_documents()
    print(f"Indexed {count} points into collection '{QDRANT_COLLECTION}'")
