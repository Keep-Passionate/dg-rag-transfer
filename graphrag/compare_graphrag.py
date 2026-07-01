"""GraphRAG 迁移试点对比：base(原题) vs dg(增强题) 在 25 篇 meta 子集上的准确率 + 配对 McNemar。

读 llm_answer_evaluator.py 产出的 llm_evaluation_results.json（零 LLM，纯统计）。
判分记录里 doc_id=PDF 文件名、method=结果文件名去 qa_results_ 前缀（此处 graphrag_base / graphrag_dg）。
必须用 (doc_id, question) 联合键并按 25 子集 PDF 名过滤——同一题文本会跨文档重复（如 "document title"）。

用法(服务器)：
  /root/miniconda3/envs/rag/bin/python compare_graphrag.py \
      --root /root/autodl-tmp/DocBench_subset \
      --eval /root/autodl-tmp/eval_graphrag/llm_evaluation_results.json
"""
import argparse
import collections
import glob
import json
import os
from math import comb
from pathlib import Path

HERE = Path(__file__).parent


def norm(q):
    return " ".join((q or "").split())


def mcnemar_exact(b, c):
    """discordant 对 (b=A对B错, c=A错B对) 的双尾精确二项检验 p 值。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("DOCBENCH_ROOT", "/root/autodl-tmp/DocBench_subset"))
    ap.add_argument("--eval", required=True, help="llm_evaluation_results.json 路径")
    ap.add_argument("--manifest", default=str(HERE / "top25.json"))
    ap.add_argument("--base-method", default="graphrag_base")
    ap.add_argument("--dg-method", default="graphrag_dg")
    a = ap.parse_args()

    manifest = json.load(open(a.manifest, encoding="utf-8"))
    docs = manifest["docs"]

    # 允许的 PDF 名集合 + 金标 (pdf名, 题) -> 题型
    allowed, gold = set(), {}
    for d in docs:
        did = d["doc_id"]
        pdfs = glob.glob(os.path.join(a.root, did, "*.pdf"))
        pdfname = os.path.basename(pdfs[0]) if pdfs else d["pdf"]
        allowed.add(pdfname)
        qa = os.path.join(a.root, did, f"{did}_qa.jsonl")
        if not os.path.exists(qa):
            continue
        for line in open(qa, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            gold[(pdfname, norm(o.get("question")))] = o.get("type", "")

    raw = json.load(open(a.eval, encoding="utf-8"))
    recs = raw.get("results", raw) if isinstance(raw, dict) else raw

    acc = collections.defaultdict(lambda: [0, 0])   # method -> [correct, total]
    verdict = {}                                     # (method, key) -> 0/1
    for r in recs:
        did = r.get("doc_id")
        if did not in allowed:
            continue
        key = (did, norm(r.get("question")))
        if gold.get(key) != "meta-data":
            continue
        try:
            av = int(r.get("accuracy"))
        except Exception:
            continue
        m = r.get("method")
        acc[m][0] += av
        acc[m][1] += 1
        verdict[(m, key)] = av

    print("==== GraphRAG 迁移 · 25 篇 meta 子集准确率 (doc_id+题 联合键) ====")
    for m in (a.base_method, a.dg_method):
        c, t = acc[m]
        print(f"  {m:14}: {c}/{t} = {c / t * 100:.1f}%" if t else f"  {m:14}: (无数据)")

    keys = [k for (mm, k) in verdict if mm == a.dg_method and (a.base_method, k) in verdict]
    b = sum(1 for k in keys if verdict[(a.base_method, k)] == 1 and verdict[(a.dg_method, k)] == 0)  # base对 dg错
    c = sum(1 for k in keys if verdict[(a.base_method, k)] == 0 and verdict[(a.dg_method, k)] == 1)  # base错 dg对
    p = mcnemar_exact(b, c)
    print(f"\n  配对 n={len(keys)}  dg净增={c - b}  (dg独对={c}, dg独错={b})")
    print(f"  McNemar 精确检验 p={p:.4g}  {'(显著 p<0.05)' if p < 0.05 else ''}")
    print("\n  期望信号：dg 准确率明显高于 base、净增为正 → meta 增益在 GraphRAG 上复现（骨干无关）。")


if __name__ == "__main__":
    main()
