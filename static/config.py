import os
PDF_DIR = "pdfs"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = "pdf_chunks"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_WORDS, CHUNK_STEP = 420, 360          # ~560 tokens, ~15% overlap (deterministic)
OCR_DPI, MIN_NATIVE_CHARS = 200, 50
TOP_DENSE, TOP_BM25, TOP_RERANK, TOP_FINAL = 30, 30, 10, 5
RERANK_MAXLEN = 256
# Any OpenAI-compatible endpoint: Groq (free tier) or Ollama
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
