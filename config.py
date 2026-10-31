import os, pathlib
# Load .env from this folder (no extra package needed)
_e = pathlib.Path(__file__).parent / ".env"
if _e.exists():
    for _ln in _e.read_text(encoding="utf-8-sig").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))
PDF_DIR = "pdfs"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = "pdf_chunks"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_WORDS, CHUNK_STEP = 420, 360          # ~560 tokens, ~15% overlap (deterministic)
OCR_DPI, MIN_NATIVE_CHARS = 200, 50
TOP_DENSE, TOP_BM25, TOP_RERANK, TOP_FINAL = 30, 30, 8, 5
CONTEXT_WORDS = int(os.getenv("CONTEXT_WORDS", "160"))  # words of each passage sent to the LLM (0 = whole chunk); keeps prompts under free-tier TPM limits
RERANK_MAXLEN = 512  # must cover a full chunk (~560 tokens) or facts late in a chunk are invisible
# Any OpenAI-compatible endpoint: Groq (free tier) or Ollama
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))  # set 0 to surface 429 rate-limit errors