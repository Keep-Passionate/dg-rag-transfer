"""Kleister-NDA: RAG-Anything ± DG 的 DG触发子集 / 未触发 / overall 对比 + McNemar（零 LLM）。

与 compare_mmlb.py 同构：读主仓 llm_answer_evaluator.py 判分产物 +
各文档 qa_results_kleister_dg.json 的 dg_used 标记。另按 kleister_key 分桶看每类字段。
用法：python compare_kleister.py [--kleister /root/autodl-tmp/Kleister_subset]
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from math import comb

ap = argparse.ArgumentParser()
ap.add_argument("--kleister", default="/root/autodl-tmp/Kleister_subset")
a = ap.parse_args()

ev_path = os.path.join(a.kleister, "_eval", "llm_evaluation_results.json")
ev = json.load(open(ev_path, encoding="utf-8"))
recs = ev.get("results", ev) if isinstance(ev, dict) else ev
norm = lambda q: " ".join((q or "").split())

# (pdf名, 题) -> DG 是否真触发 + 该题的 kleister_key（从 dg 结果文件读）
fired, keyof = {}, {}
for d in glob.glob(os.path.join(a.kleister, "k_*", "qa_results_kleister_dg.json")):
    folder = os.path.dirname(d)
    pdf = next((f for f in os.listdir(folder) if f.lower().endswith(".pdf")), None)
    doc = pdf or os.path.basename(folder)      # 无 PDF 时 doc_id 用目录名
    for r in json.load(open(d, encoding="utf-8")):
        k = (doc, norm(r.get("question")))
        fired[k] = bool(r.get("dg_used"))
        keyof[k] = r.get("kleister_key") or r.get("type", "")

acc = {}
for r in recs:
    key = (r.get("doc_id"), norm(r.get("question")))
    try:
        v = int(r.get("accuracy"))
    except Exception:
        continue
    acc.setdefault(r.get("method"), {})[key] = v


def mcn(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def line(label, keys):
    keys = [k for k in keys if k in acc.get(bm, {}) and k in acc.get(dm, {})]
    if not keys:
        print(f"  {label:22}: 无数据")
        return
    cb = sum(acc[bm][k] for k in keys)
    cd = sum(acc[dm][k] for k in keys)
    b = sum(1 for k in keys if acc[bm][k] == 1 and acc[dm][k] == 0)
    c = sum(1 for k in keys if acc[bm][k] == 0 and acc[dm][k] == 1)
    print(f"  {label:22}: n={len(keys):4d}  base {cb / len(keys) * 100:5.1f}%  "
          f"dg {cd / len(keys) * 100:5.1f}%  净增{c - b:+d}(独对{c}/独错{b})  McNemar p={mcn(b, c):.4g}")


bm, dm = "kleister_base", "kleister_dg"
print("==== Kleister-NDA · RAG-Anything ± DG ====")
line("DG触发子集", [k for k in acc.get(dm, {}) if fired.get(k) is True])
line("未触发(应≈0)", [k for k in acc.get(dm, {}) if fired.get(k) is False])
line("overall", [k for k in acc.get(dm, {}) if k in fired])
print("  ---- 按实体键分桶 ----")
by_key = defaultdict(list)
for k in acc.get(dm, {}):
    if k in fired:
        by_key[keyof.get(k, "?")].append(k)
for key in sorted(by_key):
    line(key, by_key[key])
print("\n  期望：DG触发子集 dg>base 且净增为正；未触发两列≈相等（零拖累）。")
