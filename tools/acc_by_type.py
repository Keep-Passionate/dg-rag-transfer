"""按 method × 题型统计准确率（全量）。读 eval json + DocBench 金标 type，敲定论文确切数字。

用法：python acc_by_type.py --eval /root/autodl-tmp/eval_4cond/llm_evaluation_results.json \
        --docbench /root/autodl-tmp/DocBench_subset --methods paperbase,ours,dgAllfire
"""
import argparse
import glob
import json
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True)
ap.add_argument("--docbench", default="/root/autodl-tmp/DocBench_subset")
ap.add_argument("--methods", default="", help="逗号分隔；空=全部")
a = ap.parse_args()

norm = lambda q: " ".join((q or "").split())

gold = {}
for d in glob.glob(os.path.join(a.docbench, "*")):
    if not os.path.isdir(d):
        continue
    pdfs = glob.glob(os.path.join(d, "*.pdf"))
    qa = os.path.join(d, os.path.basename(d) + "_qa.jsonl")
    if not pdfs or not os.path.exists(qa):
        continue
    pdf = os.path.basename(pdfs[0])
    for ln in open(qa, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        gold[(pdf, norm(o.get("question")))] = o.get("type", "")

raw = json.load(open(a.eval, encoding="utf-8"))
recs = raw.get("results", raw) if isinstance(raw, dict) else raw
want = {m.strip() for m in a.methods.split(",") if m.strip()}

acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in recs:
    m = r.get("method")
    if want and m not in want:
        continue
    t = gold.get((r.get("doc_id"), norm(r.get("question"))))
    if t is None:
        continue
    try:
        v = int(r.get("accuracy"))
    except Exception:
        continue
    acc[m][t][0] += v
    acc[m][t][1] += 1
    acc[m]["__overall__"][0] += v
    acc[m]["__overall__"][1] += 1

for m in sorted(acc):
    print(f"\n== {m} ==")
    for t in sorted(acc[m]):
        c, n = acc[m][t]
        label = "overall(全部)" if t == "__overall__" else (t or "(无type)")
        print(f"  {label:18} {c}/{n} = {c / n * 100:.1f}%" if n else f"  {label}: 0")
