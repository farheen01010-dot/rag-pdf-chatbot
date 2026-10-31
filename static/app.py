import json, re, time, shutil, threading, pathlib
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI
from config import *
import ingest

app = FastAPI()
S = {"payloads": [], "bm25": None}
INGEST = {}  # filename -> latest progress
tok = lambda s: re.findall(r"\w+", s.lower())

def load_index():
    qc, pts, off = ingest.client(), [], None
    if not qc.collection_exists(COLLECTION): return
    while True:
        batch, off = qc.scroll(COLLECTION, limit=1000, offset=off, with_payload=True)
        pts += [(str(p.id), p.payload) for p in batch]
        if off is None: break
    S["payloads"] = pts
    S["bm25"] = BM25Okapi([tok(p["text"]) for _, p in pts]) if pts else None

@app.on_event("startup")
def startup():
    S["rerank"] = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAXLEN)
    S["llm"] = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "none", timeout=30)
    ingest.embedder().encode(["warmup"]); S["rerank"].predict([("a", "b")])
    load_index()
    try: retrieve("warmup")
    except Exception as e: print("warmup failed:", e)

def retrieve(q):
    t, tm = time.perf_counter(), {}
    qv = ingest.embedder().encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()
    tm["embed"] = time.perf_counter() - t; t = time.perf_counter()
    dense = ingest.client().query_points(COLLECTION, query=qv, limit=TOP_DENSE, with_payload=True).points
    dmap = {str(p.id): (p.score, p.payload) for p in dense}
    scores = S["bm25"].get_scores(tok(q)) if S["bm25"] else []
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_BM25]
    bmap = {S["payloads"][i][0]: (float(scores[i]), S["payloads"][i][1]) for i in top}
    rrf = {}  # reciprocal rank fusion
    for rank, pid in enumerate(dmap): rrf[pid] = rrf.get(pid, 0) + 1 / (60 + rank)
    for rank, pid in enumerate(bmap): rrf[pid] = rrf.get(pid, 0) + 1 / (60 + rank)
    cands = sorted(rrf, key=rrf.get, reverse=True)[:TOP_RERANK]
    tm["search"] = time.perf_counter() - t; t = time.perf_counter()
    info = {pid: (dmap.get(pid) or bmap.get(pid))[1] for pid in cands}
    rr = S["rerank"].predict([(q, info[pid]["text"]) for pid in cands]) if cands else []
    order = sorted(zip(cands, rr), key=lambda x: -x[1])[:TOP_FINAL]
    tm["rerank"] = time.perf_counter() - t
    chunks = [dict(pdf=info[p]["pdf"], page_start=info[p]["page_start"], page_end=info[p]["page_end"],
                   is_ocr=info[p]["is_ocr"], text=info[p]["text"], rerank=float(s),
                   dense=dmap.get(p, (None,))[0], bm25=bmap.get(p, (None,))[0]) for p, s in order]
    return chunks, tm

SYS = ("Answer ONLY from the numbered context. Cite sources inline like [1] or [2][3]. "
       "If the context does not contain the answer, say 'Not found in the documents.' Be concise.")

class Q(BaseModel): question: str

@app.post("/chat")
def chat(q: Q):
    def gen():
        t0 = time.perf_counter()
        try: chunks, tm = retrieve(q.question)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Search failed: {e}'})}\n\n"; return
        for i, c in enumerate(chunks, 1): c["id"] = i
        yield f"data: {json.dumps({'type': 'retrieval', 'chunks': chunks, 'timings': tm})}\n\n"
        ctx = "\n\n".join(f"[{c['id']}] {c['pdf']}, p.{c['page_start']}: {c['text']}" for c in chunks)
        t1, first = time.perf_counter(), None
        try:
            stream = S["llm"].chat.completions.create(model=LLM_MODEL, stream=True, temperature=0, max_tokens=500,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q.question}"}])
            for ev in stream:
                d = ev.choices[0].delta.content
                if d:
                    first = first or time.perf_counter() - t1
                    yield f"data: {json.dumps({'type': 'token', 'text': d})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'LLM call failed: {e}'})}\n\n"
        tm.update(llm_first_token=first, total=time.perf_counter() - t0)
        yield f"data: {json.dumps({'type': 'done', 'timings': tm})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/ingest")
async def upload(files: list[UploadFile], bg: BackgroundTasks):
    for f in files:
        dest = pathlib.Path(PDF_DIR) / f.filename
        with open(dest, "wb") as o: shutil.copyfileobj(f.file, o)
        INGEST[f.filename] = {"stage": "queued"}
        bg.add_task(run_ingest, dest)
    return {"queued": [f.filename for f in files]}

def run_ingest(path):
    ingest.ingest_pdf(path, lambda **k: INGEST.__setitem__(path.name, k))
    load_index()

@app.get("/ingest/status")
def status(): return {"files": INGEST, "chunks": len(S["payloads"]),
                      "pdfs": sorted({p["pdf"] for _, p in S["payloads"]})}

@app.get("/pdf/{name}")
def pdf(name: str): return FileResponse(pathlib.Path(PDF_DIR) / pathlib.Path(name).name, media_type="application/pdf")

app.mount("/", StaticFiles(directory="static", html=True))
