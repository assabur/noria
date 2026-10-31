"""Pipeline locale en arrière-plan, prévue pour une seule instance Uvicorn."""
import logging
from threading import Lock
from rag.utils import EMBEDDINGS_MODEL

PIPELINE_LOCK = Lock()
_STATE_LOCK = Lock()
_STATE = {"status": "idle", "progress": 0, "stage": "En attente", "error": None,
          "documents": 0, "chunks": 0}


def pipeline_snapshot():
    with _STATE_LOCK:
        return dict(_STATE)


def set_pipeline(**values):
    with _STATE_LOCK:
        _STATE.update(values)


def run_pipeline(raw_dir, chunk_size, overlap, lang, embeddings_model=EMBEDDINGS_MODEL):
    try:
        from rag.ingest import ingest_documents
        from rag.index import index_documents

        output = raw_dir.parent / "chunks.parquet"
        count, documents = ingest_documents(raw_dir, output, chunk_size, overlap, lang)
        set_pipeline(progress=25, stage="Calcul des embeddings (premier chargement parfois long)",
                     documents=documents, chunks=count)
        index_documents(output, model_name=embeddings_model,
                        progress=lambda value, stage: set_pipeline(progress=value, stage=stage))
        set_pipeline(status="done", progress=100, stage="Corpus prêt pour la recherche")
    except Exception as exc:
        logging.getLogger(__name__).exception("Échec de la pipeline")
        message = str(exc) if isinstance(exc, (ValueError, UnicodeError)) else (
            "Pipeline interrompue. Vérifiez Qdrant, le téléchargement du modèle et les journaux du serveur."
        )
        set_pipeline(status="error", stage="Échec de l’indexation", error=message)
    finally:
        PIPELINE_LOCK.release()
