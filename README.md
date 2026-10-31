# RAG Chatbot over Large PDFs (open-source stack)

Ask questions about a private library of large PDFs and get streamed answers with citations (PDF name + page), a view of the retrieved passages, and a per-stage latency breakdown.

**Pipeline:** PyMuPDF (native text) + Tesseract (OCR for scanned pages) -> header/footer cleanup -> chunking with page tracking -> `bge-small-en-v1.5` embeddings -> Qdrant (HNSW) -> hybrid retrieval (dense + BM25, reciprocal rank fusion) -> cross-encoder rerank -> LLM answer with `[n]` citations.

## How it works

| Stage | What it does |
|---|---|
| Extraction | PyMuPDF reads native text. Pages with under 50 characters are rendered at 200 DPI and OCR'd with Tesseract. |
| Cleaning | Lines repeated at the top/bottom of more than 30% of pages (headers/footers) are removed; hyphenated line breaks are joined; language is detected. |
| Chunking | Word windows of 420 words (~560 tokens) with a 60-word overlap (~15%). Each chunk records its page range, so citations stay accurate when a chunk crosses pages. |
| IDs | Chunk IDs are a hash of filename + offset, so re-ingesting a PDF is idempotent and reproducible. |
| Embedding | `BAAI/bge-small-en-v1.5`, normalised, stored in Qdrant with metadata (`pdf`, `page_start`, `page_end`, `is_ocr`, `language`, `text`). |
| Index | Qdrant HNSW (m=16, ef_construct=128), cosine distance. |
| Retrieval | Top 30 dense + top 30 BM25. The top 3 of each retriever are always passed to the reranker, then the rest are filled by reciprocal rank fusion (8 candidates total). |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the candidates; the top 5 go to the LLM. |
| Generation | Each passage is trimmed to its most query-relevant ~160 words (`CONTEXT_WORDS`) to keep prompts small. The LLM must answer only from the numbered context, cite `[n]`, and say "Not found in the documents." otherwise. |
| UI | Streaming answer, clickable citations, retrieved-passages panel with dense/BM25/rerank scores, link to open the PDF at the cited page, ingestion tab with drag-and-drop upload, per-answer latency breakdown. |

## Project layout

```
rag-chatbot/
  app.py            FastAPI server: retrieval, streaming chat, upload, PDF serving
  ingest.py         PDF -> chunks -> embeddings -> Qdrant (CLI and used by the upload endpoint)
  config.py         All settings; also loads .env (no extra package needed)
  eval.py           Retrieval metrics: R@1/3/5, MRR, retrieval p95
  eval_e2e.py       End-to-end metrics against the running server
  diag.py           Shows where one question's gold passage ranks in dense/BM25/final
  eval.jsonl        30 test questions with gold PDF, page and expected answer
  static/index.html Chat UI (single file)
  pdfs/             Put your PDFs here
  .env.example      Copy to .env
  requirements.txt
```

## Setup (Windows)

Requirements: Python 3.12, Tesseract, Qdrant, and a free Groq API key (or a local Ollama model).

1. **Python 3.12:** `winget install Python.Python.3.12`, then use `py -3.12` to create the environment.
2. **Tesseract:** `winget install UB-Mannheim.TesseractOCR`. `ingest.py` also looks for `C:\Program Files\Tesseract-OCR\tesseract.exe`, so PATH is optional.
3. **Qdrant (no Docker):** download `qdrant-x86_64-pc-windows-msvc.zip` from github.com/qdrant/qdrant/releases, extract it to `C:\qdrant`, and run `C:\qdrant\qdrant.exe` in its own window. Keep that window open. (With Docker instead: `docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant`.)
4. **Environment:**
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
5. **Configuration:** copy `.env.example` to `.env` and set `LLM_API_KEY`.

macOS/Linux: the same steps apply with `brew install tesseract` / `apt install tesseract-ocr` and `source .venv/bin/activate`.

## Run

```powershell
# 1. Qdrant must be running (see above)
# 2. Put 10+ PDFs in pdfs/, then precompute embeddings (slow part: OCR on scanned pages)
python ingest.py pdfs/

# 3. Start the server
uvicorn app:app --port 8000
# open http://localhost:8000
```

You can also drag PDFs into the **Ingestion** tab while the server is running. The first start downloads the embedding and reranker models (~250 MB), so it needs internet once.

## Configuration

Set in `.env` or the environment (defaults are in `config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | none | API key for the LLM endpoint |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint (e.g. Ollama: `http://localhost:11434/v1`) |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name. Groq's available models change; list yours with the `openai` client's `models.list()` |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Use `127.0.0.1`, not `localhost` (see Troubleshooting) |
| `CONTEXT_WORDS` | `160` | Words of each passage sent to the LLM. `0` sends whole chunks |
| `LLM_MAX_RETRIES` | `2` | Set to `0` when testing to surface rate-limit errors instead of silent waits |

Retrieval sizes (`TOP_DENSE`, `TOP_BM25`, `TOP_RERANK`, `TOP_FINAL`) and chunk sizes are in `config.py`.

## Evaluation

```powershell
python eval.py eval.jsonl                 # retrieval only; stop the server first
python eval_e2e.py eval.jsonl 15          # full pipeline; needs server + Qdrant running; 15 = seconds between questions
python diag.py "question" file.pdf 37     # where does the gold page rank?
```

`eval.jsonl` has 30 questions over 10 synthetic 200-page PDFs (two with scanned pages), each with the gold PDF, page and expected answer.

Results on that test set:

| Metric | Result |
|---|---|
| Retrieval R@1 / R@3 / R@5 | 0.533 / 0.867 / 0.967 |
| MRR | 0.707 |
| Retrieval p95 | about 1.0 s |
| End-to-end answers correct | 28 / 30 (1 honest "Not found", 1 LLM request timeout) |
| Correct answers citing the gold page | 28 / 28 |
| Confident wrong answers (hallucination proxy) | 0 / 30 |
| End-to-end latency p50 / p95 | 3.6 s / 7.9 s (about 86% of answers within 5 s) |

Notes on these numbers:
- The test PDFs are synthetic: every page is similar filler text with a few planted facts, so R@1 is lower than real documents would likely give.
- "Hallucination proxy" counts answers that are neither correct nor "Not found". It is a rough check on simple factual questions, not a full faithfulness evaluation.
- The end-to-end figures were measured with context trimming on (`CONTEXT_WORDS=160`) and an 8 s pause between questions. Before trimming, the same questions gave 29/30 correct but p50 7.2 s / p95 10.3 s (measured with a 15 s pause). Re-run `eval_e2e.py` after changing `CONTEXT_WORDS`; `0` sends whole chunks.
- The one "Not found" is a retrieval miss (the right passage is outside the top 5), consistent with R@5 = 0.967. The one error was a request that timed out on the LLM provider's side.

## Latency

A single question takes about 1 s for retrieval (embed ~30 ms, search ~40 ms, rerank ~400-900 ms) plus the LLM's first token, for a total of roughly 2 s when asked in isolation. Over a 30-question run with 8 s between questions, the median was 3.6 s and the 95th percentile 7.9 s; the slowest answers come from the LLM provider (occasional multi-second delays and one timeout), not from retrieval. On Groq's free tier the key allows 8,000 tokens per minute, so asking many questions in quick succession gets throttled and answers slow down to several seconds. The trimming above and a pause between questions both help. A local model or a higher-limit key removes this limit.

## Troubleshooting

- **Page stuck on "Searching..."**: hard-refresh (`Ctrl+Shift+R`) in case the browser cached an old page, and check the server terminal for a traceback.
- **Search takes ~2 s every time (Windows)**: `localhost` can resolve to IPv6 first. Use `QDRANT_URL=http://127.0.0.1:6333`.
- **`429` / slow answers**: LLM rate limit. Wait a minute, ask fewer questions at once, or lower `CONTEXT_WORDS`.
- **LLM "model does not exist"**: the model was retired. Pick a current one for your key and set `LLM_MODEL`.
- **`ModuleNotFoundError: ingest`**: run commands from the project folder, not from `C:\qdrant`.
- **`ingest.py` prints nothing**: the `pdfs/` folder has no `.pdf` files (check for hidden double extensions like `file.pdf.pdf`).
- **"tesseract is not installed"**: install Tesseract; the code looks in `C:\Program Files\Tesseract-OCR\`.
- **Collection `pdf_chunks` doesn't exist**: ingestion has not run yet (or Qdrant was started from a different folder with a different `storage` directory).

## Known limitations

- Tables and figures are extracted as plain text; tables chunk poorly.
- Bounding boxes are not stored or highlighted; citations point to pages, not regions.
- The embedding model is English-oriented. For multilingual PDFs use `bge-m3` or `multilingual-e5` (set `EMBED_MODEL` and re-ingest).
- BM25 is built in memory at startup from Qdrant's payloads, which is fine for tens of thousands of chunks but not millions.
- Hallucination and citation checks are string-matching proxies, not a full faithfulness evaluation.
