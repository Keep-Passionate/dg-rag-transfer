"""读评测 JSON，按 method 在 meta-data 子集（与总体）报准确率 —— 填 IP&M 版 tab:baselines。

用法：python compare_baselines.py --root /root/autodl-tmp/DocBench_subset \
        --eval /root/autodl-tmp/DocBench_subset/_eval_baselines/llm_evaluation_results.json \
        [--methods longcontext,bm25,pal,paperbase,dgcore]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/root/autodl-tmp/DocBench_subset")
ap.add_argument("--eval", required=True)
ap.add_argument("--methods", default="", help="逗号分隔；空=评测里出现的全部")
a = ap.parse_args()

norm = lambda q: " ".join(str(q).split()).strip().lower()

q2type = {}
for qa in Path(a.root).glob("*/*_qa.jsonl"):
    for ln in open(qa, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            q2type[norm(d.get("question", ""))] = d.get("type", "?")

ev = json.load(open(a.eval, encoding="utf-8"))
ev = ev.get("results", ev) if isinstance(ev, dict) else ev
meta = defaultdict(lambda: [0, 0])       # method -> [对, 总]
allq = defaultdict(lambda: [0, 0])
for r in ev:
    if r.get("accuracy") is None:
        continue
    m = r.get("method", "?")
    c = int(r["accuracy"])
    allq[m][0] += c
    allq[m][1] += 1
    if q2type.get(norm(r.get("question"))) == "meta-data":
        meta[m][0] += c
        meta[m][1] += 1

methods = [x.strip() for x in a.methods.split(",") if x.strip()] or sorted(meta, key=lambda m: -(
    meta[m][0] / meta[m][1] if meta[m][1] else 0))
pct = lambda ct: f"{ct[0]}/{ct[1]} ({100 * ct[0] / ct[1]:.1f}%)" if ct[1] else "—"

print("==== DocBench 基线对比（meta-data 子集为主）====")
print(f"{'method':<16}{'meta 准确率':<22}{'overall 准确率':<22}")
print("-" * 60)
for m in methods:
    print(f"{m:<16}{pct(meta[m]):<22}{pct(allq[m]):<22}")
print("\n期望递进：base ≈ bm25 ≤ longcontext < pal < dgcore(DG-RAG)。")
