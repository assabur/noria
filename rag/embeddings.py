"""Modèles disponibles et préparation cohérente des passages et des questions."""
from rag.utils import EMBEDDINGS_MODEL

EMBEDDING_MODELS = {
    "intfloat/multilingual-e5-base": "E5 Base · multilingue",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM · multilingue, léger",
}
# Conserver la possibilité d'utiliser le modèle configuré au lancement.
EMBEDDING_MODELS.setdefault(EMBEDDINGS_MODEL, EMBEDDINGS_MODEL)
EMBEDDING_PROFILE = "retrieval-v1"


def prepare_texts(texts, model_name, role, legacy=False):
    if not legacy and model_name.startswith("intfloat/multilingual-e5-"):
        prefix = "query: " if role == "query" else "passage: "
        return [prefix + text for text in texts]
    return texts
