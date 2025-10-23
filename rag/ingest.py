from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd



def chunk_text(text: str, size: int = 300, overlap: int = 60) -> list[str]:
    chunks, step = [], max(1, size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i : i + size]
        if chunk:
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", help="Dossier contenant les .txt/.md", default="data/raw/")
    p.add_argument("--chunk-size", type=int, default=300)
    p.add_argument("--overlap", type=int, default=60)
    args = p.parse_args()

    rows = []
    raw_dir = Path(args.path)
    print(raw_dir.glob("**/*"))
    for file in raw_dir.glob("**/*"):
        if file.suffix.lower() not in {".txt", ".md"}:
                continue
        text = file.read_text(encoding="utf-8")
        print ("text")

        for i, chunk in enumerate(chunk_text(text, args.chunk_size, args.overlap)):
            rows.append({
                "doc_id": file.stem,
                "chunk_index": i,
                "text": chunk,
                "lang": "fr",
                "source": str(file),
            })

    df = pd.DataFrame(rows)
    out = Path("data/chunks.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} chunks to {out}")