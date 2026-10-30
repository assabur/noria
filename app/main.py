from pathlib import Path
import logging
import time
import uuid

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.pipeline import PIPELINE_LOCK, pipeline_snapshot, run_pipeline, set_pipeline
from rag.documents import DocumentReadError, SUPPORTED_SUFFIXES, read_document
from rag.search import answer_question, retrieve
from rag.utils import (
    EMBEDDINGS_MODEL, LANG_FILTER, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS,
    QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT,
)

app = FastAPI(title="Noria", version="0.2.0")
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class AskReq(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    lang: str | None = Field(default=None, max_length=16)
    k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def non_empty_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La question ne peut pas être vide.")
        return value


class PipelineReq(BaseModel):
    chunk_size: int = Field(default=300, ge=50, le=4000)
    overlap: int = Field(default=60, ge=0, le=1000)
    lang: str = Field(default="fr", pattern=r"^(fr|en)$")


@app.get("/", include_in_schema=False)
def interface():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def execute_query(req: AskReq, generate: bool):
    if pipeline_snapshot()["status"] == "running":
        raise HTTPException(409, "Indexation en cours : attendez la fin de la pipeline.")
    t0 = time.perf_counter()
    lang = req.lang if req.lang is not None else LANG_FILTER
    try:
        if generate:
            result = answer_question(req.question, lang=lang, k=req.k)
        else:
            docs, latency = retrieve(req.question, k=req.k, lang=lang)
            result = {"sources": docs, "retrieval_latency_ms": latency, "llm_latency_ms": 0}
    except httpx.TimeoutException as exc:
        raise HTTPException(
            504, f"Ollama n’a pas répondu dans le délai imparti ({OLLAMA_TIMEOUT_SECONDS:g} s). "
            "Le chargement du modèle et la génération sur CPU peuvent être longs. "
            "Ajustez OLLAMA_TIMEOUT_SECONDS dans .env puis redémarrez le serveur."
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Échec de l’appel Ollama. Vérifiez le service et le modèle configuré.") from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Échec de la requête RAG")
        raise HTTPException(503, "Recherche indisponible. Vérifiez Qdrant, l’index et le modèle "
                            "d’embeddings. Consultez les journaux du serveur pour le détail.") from exc
    result["total_latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return result


def run_query(req: AskReq, generate: bool):
    if not PIPELINE_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Une opération est en cours. Attendez sa fin avant de rechercher.")
    try:
        return execute_query(req, generate)
    finally:
        PIPELINE_LOCK.release()


@app.post("/ask")
def ask(req: AskReq):
    return run_query(req, generate=True)


@app.post("/retrieve")
def search(req: AskReq):
    return run_query(req, generate=False)


@app.get("/documents")
def documents():
    return {"documents": [
        {"name": str(path.relative_to(RAW_DIR)), "size": path.stat().st_size}
        for path in sorted(RAW_DIR.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]}


@app.post("/documents", status_code=201)
async def upload_document(request: Request, filename: str):
    # Corps binaire envoyé par le navigateur : pas de dépendance multipart supplémentaire.
    if Path(filename).name != filename or "\\" in filename or len(filename) > 180:
        raise HTTPException(400, "Nom de fichier invalide.")
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, "Formats acceptés : .txt et .md en UTF-8, et .pdf avec texte extractible.")
    content = bytearray()
    async for block in request.stream():
        content.extend(block)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Le fichier dépasse la limite de 5 Mo.")
    try:
        pages = await run_in_threadpool(read_document, filename, bytes(content))
        if not any(text.strip() for _, text in pages):
            raise DocumentReadError(f"{filename} : le document est vide.")
    except DocumentReadError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not PIPELINE_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Attendez la fin de l’indexation avant d’importer un fichier.")
    try:
        directory = RAW_DIR / "uploads" / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_bytes(content)
    finally:
        PIPELINE_LOCK.release()
    return {"name": str(path.relative_to(RAW_DIR)), "size": len(content),
            "pages": len(pages) if Path(filename).suffix.lower() == ".pdf" else None}


@app.get("/pipeline")
def pipeline_status():
    return pipeline_snapshot()


@app.post("/pipeline", status_code=202)
def start_pipeline(req: PipelineReq, background: BackgroundTasks):
    if req.overlap >= req.chunk_size:
        raise HTTPException(422, "Le chevauchement doit être inférieur à la taille des extraits.")
    if not PIPELINE_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Une pipeline est déjà en cours.")
    set_pipeline(status="running", progress=5, stage="Préparation des documents", error=None,
                 documents=0, chunks=0)
    background.add_task(run_pipeline, RAW_DIR, req.chunk_size, req.overlap, req.lang)
    return pipeline_snapshot()


@app.get("/status")
def status():
    services = {}
    with httpx.Client(timeout=3.0) as client:
        for name, url in (
            ("qdrant", f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}"),
            ("ollama", f"{OLLAMA_BASE_URL}/api/tags"),
        ):
            try:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
                if name == "qdrant":
                    count = payload["result"]["points_count"]
                    services[name] = {"ok": count > 0, "detail": f"{count} extraits indexés" if count else "Collection vide"}
                else:
                    names = {model["name"] for model in payload.get("models", [])}
                    available = OLLAMA_MODEL in names or f"{OLLAMA_MODEL}:latest" in names
                    services[name] = {"ok": available, "detail": OLLAMA_MODEL if available else f"Modèle à télécharger : {OLLAMA_MODEL}"}
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                services[name] = {"ok": False, "detail": "Collection absente ou service indisponible" if name == "qdrant" else "Service indisponible"}
    return {"services": services, "config": {"model": OLLAMA_MODEL, "embeddings": EMBEDDINGS_MODEL,
            "collection": QDRANT_COLLECTION, "lang": LANG_FILTER}}


@app.get("/metrics")
def metrics():
    return {"uptime": "unknown", "notes": "add Prometheus later"}
