"""End-to-end eval against the RUNNING server (uvicorn + Qdrant up).
Usage: python eval_e2e.py eval.jsonl [pause_seconds] [max_questions]"""
import sys, json, re, time, urllib.request as u
import numpy as np
URL = "http://127.0.0.1:8000/chat"
PAUSE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0   # seconds between questions (free-tier rate limits)
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None   # optional: only first N questions

def norm(s):
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s).lower()
    return re.sub(r"(?<=\d)[,\u00a0\u2007\u2009\u202f](?=\d)", "", s)
def safe(s): return s.encode("ascii", "replace").decode()

def ask(q):
    t0, ttft, chunks, text, err = time.perf_counter(), None, [], "", None
    req = u.Request(URL, json.dumps({"question": q}).encode(), {"Content-Type": "application/json"})
    try:
        with u.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: "): continue
                m = json.loads(line[6:])
                if m["type"] == "retrieval": chunks = m["chunks"]
                elif m["type"] == "token":
                    ttft = ttft or time.perf_counter() - t0; text += m["text"]
                elif m["type"] == "error": err = m["message"]
    except Exception as e: err = str(e)
    return text, chunks, ttft, time.perf_counter() - t0, err

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")][:LIMIT]
tot, ttf, R = [], [], dict(correct=0, notfound=0, wrong=0, errors=0, cited_ok=0)
for n, r in enumerate(rows, 1):
    text, chunks, ttft, total, err = ask(r["q"])
    if err and not text: R["errors"] += 1; print(f"[{n}] ERROR {safe(err)[:500]}"); time.sleep(PAUSE); continue
    tot.append(total); ttf.append(ttft or total)
    nt = norm(text)
    correct = all(norm(a) in nt for a in r["answer"])
    nf = "not found in the documents" in nt
    cited = {int(x) for x in re.findall(r"[\[【［](\d+)[^\]】］]*[\]】］]", text)}
    gold_cited = any(c["id"] in cited and c["pdf"] == r["pdf"] and c["page_start"] <= r["page"] <= c["page_end"] for c in chunks)
    if correct:
        R["correct"] += 1; R["cited_ok"] += gold_cited
    elif nf: R["notfound"] += 1
    else: R["wrong"] += 1
    tag = "ok" if correct else ("not-found" if nf else "WRONG")
    print(f"[{n}] {tag:9} cite={'y' if gold_cited else 'n'} total={total:.2f}s  {safe(r['q'])[:60]}")
    if not correct: print("     ->", safe(text.strip())[:160])
    elif not gold_cited: print(f"     cited={sorted(cited)} gold_chunk_ids={[c['id'] for c in chunks if c['pdf']==r['pdf'] and c['page_start']<=r['page']<=c['page_end']]} | {safe(text.strip())[:140]}")
    time.sleep(PAUSE)
n = len(rows)
print("\n=== End-to-end summary ===")
print(f"answer correct        : {R['correct']}/{n} ({R['correct']/n:.0%})")
print(f"'not found' (misses)  : {R['notfound']}/{n}")
print(f"confident but wrong   : {R['wrong']}/{n}  <- hallucination proxy ({R['wrong']/n:.0%})")
print(f"citation accuracy     : {R['cited_ok']}/{max(R['correct'],1)} of correct answers cite the gold page")
print(f"request errors        : {R['errors']}")
if tot: print(f"latency total  p50/p95: {np.percentile(tot,50):.2f}s / {np.percentile(tot,95):.2f}s | first token p95: {np.percentile(ttf,95):.2f}s")