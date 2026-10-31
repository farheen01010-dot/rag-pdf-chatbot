"""python eval.py eval.jsonl   (lines: {"q": "...", "pdf": "file.pdf", "page": 45})"""
import sys, json, time, numpy as np, app
app.startup()
rows = [json.loads(l) for l in open(sys.argv[1])]
hits, rr, lat = {1: 0, 3: 0, 5: 0}, [], []
for r in rows:
    t = time.perf_counter(); chunks, _ = app.retrieve(r["q"]); lat.append(time.perf_counter() - t)
    rank = next((i for i, c in enumerate(chunks, 1)
                 if c["pdf"] == r["pdf"] and c["page_start"] <= r["page"] <= c["page_end"]), None)
    rr.append(1 / rank if rank else 0)
    for k in hits: hits[k] += bool(rank and rank <= k)
n = len(rows)
print({f"R@{k}": round(v / n, 3) for k, v in hits.items()}, "MRR", round(sum(rr) / n, 3),
      "retrieval p95 (s)", round(float(np.percentile(lat, 95)), 3))
