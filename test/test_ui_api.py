"""Tests hors ligne : API réelle, Parquet et Qdrant en mémoire ; embeddings/LLM simulés."""
import hashlib
from io import BytesIO

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app import main, pipeline
from rag import index, search


class TestEmbedder:
    __test__ = False

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            vector = np.zeros(64)
            for word in text.lower().split():
                slot = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % len(vector)
                vector[slot] += 1
            vectors.append(vector)
        return np.array(vectors)


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(main, "RAW_DIR", raw)
    memory = QdrantClient(":memory:")
    embedder = TestEmbedder()
    monkeypatch.setattr(search, "get_client", lambda: memory)
    monkeypatch.setattr(search, "get_embedder", lambda: embedder)
    monkeypatch.setattr(index, "get_client", lambda: memory)
    monkeypatch.setattr(index, "get_embedder", lambda: embedder)
    pipeline.set_pipeline(status="idle", stage="En attente", progress=0, error=None, chunks=0, documents=0)
    with TestClient(main.app) as test_client:
        yield test_client
    memory.close()


def upload(client, name="notes.txt", text="saturne anneaux planète"):
    return client.post("/documents", params={"filename": name}, content=text.encode("utf-8"))


def build(client, **kwargs):
    response = client.post("/pipeline", json=kwargs)
    assert response.status_code == 202
    assert client.get("/pipeline").json()["status"] == "done"


def test_interface_and_assets_without_models(client):
    assert "Noria" in client.get("/").text
    for asset in ("app.js", "style.css"):
        assert client.get(f"/static/{asset}").status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_index_and_retrieve_end_to_end(client):
    assert upload(client).status_code == 201
    assert len(client.get("/documents").json()["documents"]) == 1
    build(client, lang="fr")
    result = client.post("/retrieve", json={"question": "saturne anneaux planète", "lang": "fr", "k": 1})
    assert result.status_code == 200
    source = result.json()["sources"][0]
    assert source["text"] == "saturne anneaux planète"
    assert source["correspondence_pct"] == 100.0
    assert "uploads/" in source["source"]
    assert result.json()["llm_latency_ms"] == 0
    assert client.post("/retrieve", json={"question": "saturne", "lang": "en"}).json()["sources"] == []
    assert client.post("/retrieve", json={"question": "saturne", "lang": ""}).json()["sources"]


def test_generation_receives_context_and_returns_sources(client, monkeypatch):
    upload(client)
    build(client)
    prompts = []
    def llm(prompt):
        prompts.append(prompt)
        return "Réponse de test", 12
    monkeypatch.setattr(search, "call_ollama", llm)
    result = client.post("/ask", json={"question": "saturne", "lang": "fr"}).json()
    assert result["answer"] == "Réponse de test"
    assert "saturne anneaux planète" in prompts[0]
    assert result["sources"][0]["text"]
    assert result["llm_latency_ms"] == 12


def test_no_generation_without_matching_documents(client, monkeypatch):
    upload(client)
    build(client)
    def unexpected_call(prompt):
        pytest.fail("Ollama ne doit pas être appelé sans contexte")
    monkeypatch.setattr(search, "call_ollama", unexpected_call)
    result = client.post("/ask", json={"question": "saturne", "lang": "en"}).json()
    assert result["sources"] == []
    assert result["llm_latency_ms"] == 0


def test_identical_filenames_are_preserved_and_reindex_replaces(client):
    upload(client, text="premier document")
    upload(client, text="second document")
    documents = client.get("/documents").json()["documents"]
    assert len(documents) == 2
    assert documents[0]["name"] != documents[1]["name"]
    build(client)
    first = client.post("/retrieve", json={"question": "document", "lang": "fr"}).json()
    assert len({source["doc_id"] for source in first["sources"]}) == 2
    build(client, lang="en")
    assert client.post("/retrieve", json={"question": "document", "lang": "fr"}).json()["sources"] == []


@pytest.mark.parametrize("filename,content,code", [
    ("../escape.txt", b"text", 400), ("bad\\file.txt", b"text", 400),
    ("file.docx", b"docx", 415), ("file.pdf", b"pdf", 400), ("empty.txt", b"  ", 400),
    ("bad.txt", b"\xff", 400), ("large.txt", b"a" * (5 * 1024 * 1024 + 1), 413),
])
def test_invalid_uploads(client, filename, content, code):
    assert client.post("/documents", params={"filename": filename}, content=content).status_code == code
    assert client.get("/documents").json()["documents"] == []


@pytest.mark.parametrize("body", [{"question": "  "}, {"question": "ok", "k": 0}, {"question": "ok", "k": 11}])
def test_invalid_query(client, body):
    assert client.post("/retrieve", json=body).status_code == 422


def test_pipeline_validation_empty_corpus_and_lock(client):
    assert client.post("/pipeline", json={"chunk_size": 50, "overlap": 50}).status_code == 422
    client.post("/pipeline", json={})
    assert client.get("/pipeline").json()["status"] == "error"
    assert not pipeline.PIPELINE_LOCK.locked()
    pipeline.PIPELINE_LOCK.acquire()
    try:
        assert client.post("/pipeline", json={}).status_code == 409
        assert upload(client).status_code == 409
        assert client.post("/retrieve", json={"question": "test"}).status_code == 409
    finally:
        pipeline.PIPELINE_LOCK.release()


def test_default_language_follows_config(client, monkeypatch):
    upload(client)
    build(client, lang="fr")
    monkeypatch.setattr(main, "LANG_FILTER", "fr")
    result = client.post("/retrieve", json={"question": "saturne"}).json()
    assert len(result["sources"]) == 1


def test_generation_timeout_is_actionable(client, monkeypatch):
    upload(client)
    build(client)
    def timeout(prompt):
        raise httpx.ReadTimeout("timeout")
    monkeypatch.setattr(search, "call_ollama", timeout)
    response = client.post("/ask", json={"question": "saturne", "lang": "fr"})
    assert response.status_code == 504
    assert "OLLAMA_TIMEOUT_SECONDS" in response.json()["detail"]
    assert not pipeline.PIPELINE_LOCK.locked()


def test_ollama_uses_configured_generation_timeout(monkeypatch):
    real_client = httpx.Client
    monkeypatch.setattr(search, "OLLAMA_TIMEOUT_SECONDS", 420.0)

    def respond(request):
        # Vérifie les délais réellement transmis au transport HTTP.
        assert request.extensions["timeout"]["read"] == 420.0
        assert request.extensions["timeout"]["connect"] == 5.0
        return httpx.Response(200, json={"response": "Réponse lente"})

    def client_with_transport(**kwargs):
        return real_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(search.httpx, "Client", client_with_transport)
    answer, latency = search.call_ollama("Question de test")
    assert answer == "Réponse lente"
    assert latency >= 0


@pytest.mark.parametrize("score,expected", [(-0.4, 0), (0.856, 85.6), (1.00001, 100)])
def test_correspondence_is_clamped_cosine(score, expected):
    assert search.correspondence_percent(score) == expected


def pdf_bytes(pages, password=None):
    """Vrai PDF généré en mémoire ; aucune simulation de l'extraction pypdf."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=595, height=842)
        if text:
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
            })
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 40 780 Td ({escaped}) Tj ET".encode("latin-1"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password, algorithm="RC4-128")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_upload_pipeline_and_page_sources(client):
    content = pdf_bytes(["Boreal autonomie 18 heures", "", "Garantie 24 mois"])
    response = client.post("/documents", params={"filename": "produit.PDF"}, content=content)
    assert response.status_code == 201
    assert response.json()["pages"] == 3
    assert client.get("/documents").json()["documents"][0]["name"].endswith(".PDF")
    upload(client, text="Document annexe texte brut")
    build(client)
    result = client.post("/retrieve", json={"question": "Garantie 24 mois", "lang": "fr", "k": 10}).json()
    assert result["sources"][0]["page"] == 3
    assert "Garantie 24 mois" in result["sources"][0]["text"]
    assert result["sources"][0]["correspondence_pct"] == 100
    assert {source["page"] for source in result["sources"]} == {None, 1, 3}
    assert "page 3]" in search.build_prompt("Garantie ?", [result["sources"][0]])


@pytest.mark.parametrize("content,detail", [
    (b"%PDF-1.7\ninvalid", "invalide"),
    (pdf_bytes([""]), "OCR"),
    (pdf_bytes(["Texte secret"], password="secret"), "protégé"),
])
def test_unusable_pdf_is_rejected_before_storage(client, content, detail):
    response = client.post("/documents", params={"filename": "document.pdf"}, content=content)
    assert response.status_code == 400
    assert detail in response.json()["detail"]
    assert client.get("/documents").json()["documents"] == []


def test_bad_pdf_added_manually_does_not_replace_existing_index(client):
    upload(client)
    build(client)
    (main.RAW_DIR / "scan.pdf").write_bytes(pdf_bytes([""]))
    client.post("/pipeline", json={})
    status = client.get("/pipeline").json()
    assert status["status"] == "error"
    assert "scan.pdf" in status["error"] and "OCR" in status["error"]
    assert not pipeline.PIPELINE_LOCK.locked()
    result = client.post("/retrieve", json={"question": "saturne", "lang": "fr"}).json()
    assert result["sources"]
