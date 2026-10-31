from __future__ import annotations
import argparse
from pathlib import Path

from rag.documents import SUPPORTED_SUFFIXES, read_document


def chunk_text(text: str, size: int = 300, overlap: int = 60) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("La taille doit être positive et le chevauchement compris entre 0 et taille - 1.")
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]


def ingest_documents(raw_dir, output, size=300, overlap=60, lang="multi"):
    import pandas as pd

    rows = []
    documents = 0
    raw_dir = Path(raw_dir)
    for file in sorted(raw_dir.rglob("*")):
        if not file.is_file() or file.is_symlink() or file.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        pages = read_document(file.name, file.read_bytes())
        chunk_index = 0
        for page, text in pages:
            for chunk in chunk_text(text, size, overlap):
                rows.append({"doc_id": file.relative_to(raw_dir).as_posix(),
                             "chunk_index": chunk_index, "page": page,
                             "text": chunk, "lang": lang, "source": str(file)})
                chunk_index += 1
        if chunk_index:
            documents += 1
    if not rows:
        raise ValueError("Aucun texte exploitable. Importez un fichier .txt, .md ou un PDF avec du texte.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).astype({"page": "Int64"}).to_parquet(output, index=False)
    return len(rows), documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/raw/")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--lang", default="multi", choices=["multi", "fr", "en"])
    args = parser.parse_args()
    count, _ = ingest_documents(args.path, "data/chunks.parquet", args.chunk_size, args.overlap, args.lang)
    print(f"Saved {count} chunks to data/chunks.parquet")
