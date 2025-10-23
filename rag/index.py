from __future__ import annotations
import pandas as pd
import uuid
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from  utils import (
QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDINGS_MODEL
)


DF_PATH = "data/chunks.parquet"


client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBEDDINGS_MODEL)


# Création/Reset de la collection
def ensure_collection(dim: int):
    client.recreate_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )

if __name__ == "__main__":
    df = pd.read_parquet(DF_PATH)
    vectors = embedder.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)
    ensure_collection(vectors.shape[1])

    points = []

    for row, vec in zip(df.to_dict("records"), vectors):
        # UUID v5 déterministe à partir de doc_id#chunk_index
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{row['doc_id']}#{row['chunk_index']}"))
        points.append(models.PointStruct(
            id=pid,                          # <- UUID au format string
            vector=vec.tolist(),
            payload={
                "doc_id": row["doc_id"],
                "chunk_index": int(row["chunk_index"]),
                "lang": row["lang"],
                "source": row["source"],
                "text": row["text"],
            }
        ))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    # Index de payload utiles
    for field, schema in {
        "lang": models.PayloadSchemaType.KEYWORD,
        "doc_id": models.PayloadSchemaType.KEYWORD,
    }.items():
        try:
            client.create_payload_index(QDRANT_COLLECTION, field_name=field, field_schema=schema)
        except Exception:
             pass
    print(f"Indexed {len(points)} points into collection '{QDRANT_COLLECTION}'")


