"""PDF -> clean text (native/OCR) -> chunks -> embeddings -> Qdrant. Usage: python ingest.py pdfs/"""
import io, re, sys, uuid, pathlib, collections
from concurrent.futures import ProcessPoolExecutor
import fitz, pytesseract
from PIL import Image
from langdetect import detect
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from config import *
import os
_T = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_T): pytesseract.pytesseract.tesseract_cmd = _T

def _norm(line): return re.sub(r"\d+", "#", line.strip().lower())

def extract_pdf(path):
    """Returns [(page_no, text, is_ocr)] with headers/footers stripped."""
    doc, raw = fitz.open(path), []
    for i, p in enumerate(doc, 1):
        t, ocr = p.get_text("text").strip(), False
        if len(t) < MIN_NATIVE_CHARS:  # scanned page -> OCR
            try:
                img = Image.open(io.BytesIO(p.get_pixmap(dpi=OCR_DPI).tobytes("png")))
                t, ocr = pytesseract.image_to_string(img).strip(), True
            except Exception: pass
        raw.append((i, t.split("\n"), ocr))
    # repeated first/last 2 lines on >30% of pages = header/footer
    cnt = collections.Counter(_norm(l) for _, ls, _ in raw for l in ls[:2] + ls[-2:] if l.strip())
    bad = {k for k, v in cnt.items() if v > 0.3 * len(raw)}
    out = []
    for i, ls, ocr in raw:
        keep = [l for j, l in enumerate(ls) if not (_norm(l) in bad and (j < 2 or j >= len(ls) - 2))]
        txt = re.sub(r"-\n(\w)", r"\1", "\n".join(keep))
        out.append((i, re.sub(r"\s+", " ", txt).strip(), ocr))
    return out

def chunk_pages(pages):
    """Word-window chunks that remember which pages they span."""
    words = [(w, pg, ocr) for pg, t, ocr in pages for w in t.split()]
    for s in range(0, max(len(words), 1), CHUNK_STEP):
        win = words[s:s + CHUNK_WORDS]
        if len(win) < 40 and s > 0: break
        yield s, " ".join(w for w, _, _ in win), win[0][1], win[-1][1], any(o for _, _, o in win)

_model = None
def embedder():
    global _model
    _model = _model or SentenceTransformer(EMBED_MODEL)
    return _model

_qc = None
def client():
    global _qc
    _qc = _qc or QdrantClient(url=QDRANT_URL)
    return _qc

def ensure_collection(qc):
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(COLLECTION,
            vectors_config=models.VectorParams(size=embedder().get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128))  # HNSW ANN index

def ingest_pdf(path, progress=lambda **k: None):
    path = pathlib.Path(path); qc = client(); ensure_collection(qc)
    progress(stage="extracting", file=path.name)
    with ProcessPoolExecutor(1) as ex: pages = ex.submit(extract_pdf, str(path)).result()
    ocr_n = sum(o for _, _, o in pages)
    chunks = list(chunk_pages(pages))
    progress(stage="embedding", file=path.name, pages=len(pages), ocr_pages=ocr_n, chunks=len(chunks))
    texts = [c[1] for c in chunks]
    vecs = embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    try: lang = detect(" ".join(texts[:5]))
    except Exception: lang = "unknown"
    pts = [models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{off}")),  # deterministic -> idempotent re-ingest
        vector=v.tolist(),
        payload=dict(pdf=path.name, page_start=ps, page_end=pe, is_ocr=o, language=lang, text=t))
        for (off, t, ps, pe, o), v in zip(chunks, vecs)]
    for i in range(0, len(pts), 256): qc.upsert(COLLECTION, pts[i:i + 256])
    progress(stage="done", file=path.name, pages=len(pages), ocr_pages=ocr_n, chunks=len(chunks))
    return len(chunks)

if __name__ == "__main__":
    for f in sorted(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else PDF_DIR).glob("*.pdf")):
        print(f.name, ingest_pdf(f, lambda **k: print("  ", k)), "chunks")
