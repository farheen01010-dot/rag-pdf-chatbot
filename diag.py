"""Where does the gold passage rank? Usage: python diag.py "question" file.pdf page"""
import sys, app, ingest
from config import COLLECTION, QUERY_PREFIX
app.startup()
q, pdf, page = sys.argv[1], sys.argv[2], int(sys.argv[3])
gold = {pid: (p["page_start"], p["page_end"]) for pid, p in app.S["payloads"]
        if p["pdf"] == pdf and p["page_start"] <= page <= p["page_end"]}
print("gold chunks (page ranges):", list(gold.values()))
qv = ingest.embedder().encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()
dense = ingest.client().query_points(COLLECTION, query=qv, limit=300).points
dr = {str(p.id): i + 1 for i, p in enumerate(dense)}
sc = app.S["bm25"].get_scores(app.tok(q))
order = sorted(range(len(sc)), key=lambda i: -sc[i])
br = {app.S["payloads"][i][0]: r + 1 for r, i in enumerate(order)}
for g, pr in gold.items(): print(f"  chunk {pr}: dense rank {dr.get(g)}, bm25 rank {br.get(g)}")
chunks, tm = app.retrieve(q)
print("final top-5:", [(c["pdf"][:18], c["page_start"], c["page_end"], round(c["rerank"], 2)) for c in chunks])
