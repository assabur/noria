"""Lecture des documents, partagée par l'import et l'ingestion."""
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class DocumentReadError(ValueError):
    """Document inutilisable pour la recherche textuelle."""


def read_document(filename: str, content: bytes) -> list[tuple[int | None, str]]:
    """Renvoie les pages (numérotées à partir de 1), ou un texte sans page."""
    if Path(filename).suffix.lower() != ".pdf":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentReadError(f"{filename} : le texte doit être encodé en UTF-8.") from exc
        return [(None, text)]

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise DocumentReadError(
                f"{filename} : PDF protégé par chiffrement. Importez une copie non protégée."
            )
        pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    except DocumentReadError:
        raise
    except Exception as exc:
        raise DocumentReadError(
            f"{filename} : PDF invalide ou illisible. Essayez de le réexporter en PDF."
        ) from exc
    if not any(text.strip() for _, text in pages):
        raise DocumentReadError(
            f"{filename} : aucun texte extractible. Ce PDF est vide ou scanné ; "
            "effectuez un OCR avant de l’importer."
        )
    return pages
