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
    ap.add_argument("--all-docs", action="store_true", help="统计 DocBench_subset 全部文档(全量)")
    ap.add_argument("--base-method", default="graphrag_base")
    ap.add_argument("--dg-method", default="graphrag_dg")
    a = ap.parse_args()

    if a.all_docs:
        docs = [{"doc_id": p.name} for p in sorted(Path(a.root).iterdir())
                if p.is_dir() and (p / f"{p.name}_qa.jsonl").exists()]
    else:
        docs = json.load(open(a.manifest, encoding="utf-8"))["docs"]

    # 允许的 PDF 名集合 + 金标 (pdf名, 题) -> 题型
    allowed, gold = set(), {}
    for d in docs:
        did = d["doc_id"]
        pdfs = glob.glob(os.path.join(a.root, did, "*.pdf"))
        pdfname = os.path.basename(pdfs[0]) if pdfs else d.get("pdf")
        if not pdfname:
            continue
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

    # method -> type -> [correct, total]；verdict[(method, key)] 供配对 McNemar
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    verdict = collections.defaultdict(dict)          # method -> key -> 0/1
    vtype = {}                                        # key -> type
    for r in recs:
        did = r.get("doc_id")
        if did not in allowed:
            continue
        key = (did, norm(r.get("question")))
        t = gold.get(key)
        if t is None:                                # 不在子集金标里
            continue
        try:
            av = int(r.get("accuracy"))
        except Exception:
            continue
        m = r.get("method")
        acc[m][t][0] += av
        acc[m][t][1] += 1
        acc[m]["__overall__"][0] += av
        acc[m]["__overall__"][1] += 1
        verdict[m][key] = av
        vtype[key] = t

    bm, dm = a.base_method, a.dg_method
    types = sorted(t for t in set(list(acc[bm]) + list(acc[dm])) if t != "__overall__")

    def pct(m, t):
        c, n = acc[m].get(t, [0, 0])
        return f"{c}/{n}={c / n * 100:.1f}%" if n else "(无)"

    scope = "全量" if a.all_docs else "子集"
    print(f"==== 迁移对比 [{bm} vs {dm}] · {scope} · 分题型准确率 ====")
    print(f"  {'题型':<14}{bm:>18}{dm:>18}   Δ")
    for t in types + ["__overall__"]:
        bc, bn = acc[bm].get(t, [0, 0])
        dc, dn = acc[dm].get(t, [0, 0])
        d = (dc / dn * 100 - bc / bn * 100) if bn and dn else 0.0
        label = "overall(全部)" if t == "__overall__" else t
        star = "  ← 只应此行大涨" if t == "meta-data" else ("  ← 非meta应≈0(不拖累)" if t != "__overall__" else "")
        print(f"  {label:<14}{pct(bm, t):>18}{pct(dm, t):>18}  {d:+5.1f}pp{star}")

    # 配对 McNemar：meta 口径 + overall 口径
    def mcnemar(only_type=None):
        keys = [k for k in verdict[dm] if k in verdict[bm] and (only_type is None or vtype.get(k) == only_type)]
        b = sum(1 for k in keys if verdict[bm][k] == 1 and verdict[dm][k] == 0)
        c = sum(1 for k in keys if verdict[bm][k] == 0 and verdict[dm][k] == 1)
        return len(keys), b, c, mcnemar_exact(b, c)

    for label, ot in [("meta-data", "meta-data"), ("overall", None)]:
        n, b, c, p = mcnemar(ot)
        sig = "(显著 p<0.05)" if p < 0.05 else ""
        print(f"\n  [{label}] 配对 n={n}  dg净增={c - b}  (dg独对={c}, dg独错={b})  McNemar p={p:.4g} {sig}")
    print("\n  期望：meta 行大涨且显著；非 meta 各行 Δ≈0（DG 弃权→恒等回退→不拖累）；overall 只升不降。")


if __name__ == "__main__":
    main()
