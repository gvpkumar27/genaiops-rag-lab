from pathlib import Path
import re
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.ops.metrics import set_docs_indexed_total, set_qdrant_collection_points
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts


def _merge_fragmented_letters(text: str) -> str:
    # Merge OCR/PDF fragments like:
    # "H T T P", "R a n ge", "R e qu e s t s" -> "HTTP", "Range", "Requests"
    pattern = re.compile(r"\b(?:[A-Za-z]{1,2}\s+){3,}[A-Za-z]{1,12}\b")

    def repl(match: re.Match[str]) -> str:
        tokens = match.group(0).split()
        single_count = sum(1 for t in tokens if len(t) == 1)
        if single_count >= 2:
            return "".join(tokens)
        return match.group(0)

    prev = None
    out = text
    # Apply a few passes because fixing one sequence can expose another nearby.
    for _ in range(3):
        if out == prev:
            break
        prev = out
        out = pattern.sub(repl, out)
    return out


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _merge_fragmented_letters(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        raw = "\n".join(page.extract_text() or "" for page in reader.pages)
        return _normalize_text(raw)

    raw = path.read_text(encoding="utf-8", errors="ignore")
    return _normalize_text(raw)


def ensure_collection(client: QdrantClient, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION in existing:
        client.delete_collection(collection_name=settings.QDRANT_COLLECTION)

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def main():
    docs_dir = Path("data/docs")
    files = [p for p in docs_dir.rglob("*") if p.is_file()]
    if not files:
        print("Add docs into data/docs first (PDF/TXT/MD).")
        return

    client = QdrantClient(url=settings.QDRANT_URL)

    test_vec = embed_texts(["dimension check"])[0]
    ensure_collection(client, dim=len(test_vec))

    points = []
    pid = 1

    for fp in files:
        text = read_file(fp)
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            continue

        vecs = embed_texts(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            payload = {
                "source": str(fp).replace("\\", "/"),
                "chunk_id": i,
                "text": chunk,
                "type": fp.suffix.lower().lstrip(".") or "txt",
            }
            points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))
            pid += 1

    for i in tqdm(range(0, len(points), 64), desc="Upserting to Qdrant"):
        client.upsert(settings.QDRANT_COLLECTION, points[i : i + 64])

    set_docs_indexed_total(len(files))
    set_qdrant_collection_points(len(points))
    print(f"OK: Ingested {len(points)} chunks into '{settings.QDRANT_COLLECTION}'")


if __name__ == "__main__":
    main()
