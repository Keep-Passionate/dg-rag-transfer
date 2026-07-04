"""诊断 contract_date 抽取器：逐篇打印 [金标 vs 抽取值 vs base答 vs dg答]（零 API）。

回答"109 注入却 +0"到底是①抽错日期 ②抽对但模型没用 ③base本来就对。
用法：python diag_contract_date.py --limit 25
"""
import argparse
import glob
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--kleister", default="/root/autodl-tmp/Kleister_subset")
ap.add_argument("--dg-core", default="/root/autodl-tmp/rag-L1/reproduce")
ap.add_argument("--limit", type=int, default=25)
a = ap.parse_args()

os.environ["DG_CONTRACT_DATE"] = "true"           # 开启 extensibility 算子
sys.path.insert(0, a.dg_core)
import dg_core                                     # noqa: E402

norm = lambda s: " ".join((s or "").split())


def ans_of(folder, tag, question):
    f = os.path.join(folder, f"qa_results_kleister_{tag}.json")
    if not os.path.exists(f):
        return None
    for r in json.load(open(f, encoding="utf-8")):
        if norm(r.get("question")) == norm(question):
            return r.get("answer")
    return None


def year(s):
    import re
    m = re.search(r"\b(19|20)\d{2}\b", str(s) or "")
    return m.group(0) if m else None


n = extracted_n = year_match = base_ok_year = 0
for folder in sorted(glob.glob(os.path.join(a.kleister, "k_*"))):
    qa = os.path.join(folder, os.path.basename(folder) + "_qa.jsonl")
    if not os.path.exists(qa):
        continue
    rows = [json.loads(l) for l in open(qa, encoding="utf-8") if l.strip()]
    ed = next((r for r in rows if r.get("kleister_key") == "effective_date"), None)
    if not ed:
        continue
    pdf = next((os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(".pdf")), None)
    if not pdf:
        continue
    q, gold = ed["question"], ed.get("answer", "")
    fact = dg_core.ground(q, pdf)
    got = fact.value if (fact and fact.kind == "contract_date") else None
    base_a = ans_of(folder, "base", q)
    dg_a = ans_of(folder, "dg", q)
    n += 1
    extracted_n += bool(got)
    ym = year(got) and year(got) == year(gold)
    year_match += bool(ym)
    base_ok_year += bool(year(base_a) and year(base_a) == year(gold))
    if n <= a.limit:
        print(f"[{os.path.basename(folder)}] gold={gold!r}")
        print(f"    抽取={got!r}  (kind={fact.kind if fact else None}, conf={round(fact.confidence,2) if fact else None})")
        print(f"    base答={str(base_a)[:80]!r}")
        print(f"    dg答  ={str(dg_a)[:80]!r}")

print("\n================ 汇总 ================")
print(f"effective_date 文档 {n} 篇")
print(f"  抽取器产出日期(非空): {extracted_n}/{n}")
print(f"  抽取年份==金标年份 : {year_match}/{n}   <- 抽取器是否抽对(粗看年份)")
print(f"  base答年份==金标年份: {base_ok_year}/{n}   <- base 自己对不对(headroom)")
