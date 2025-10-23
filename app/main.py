from fastapi import FastAPI
from pydantic import BaseModel
import time
from rag.search import answer_question


app = FastAPI(title="RAG Local", version="0.1.0")


class AskReq(BaseModel):
    question: str
    lang: str | None = None

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskReq):
    t0 = time.time()
    result = answer_question(req.question, lang=req.lang or "en")
    result["total_latency_ms"] = int((time.time() - t0) * 1000)
    return result

@app.get("/metrics")
def metrics():
    # Expose quelques compteurs simples; à enrichir selon besoin
    return {"uptime": "unknown", "notes": "add Prometheus later"}