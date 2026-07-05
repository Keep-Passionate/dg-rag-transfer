"""PDFTriage（Adobe, EMNLP'24）探针：dg_core.parse 可作用子集统计（零 API）。

PDFTriage 的题带 category（Structure / Extraction / Figure / Table / Text / ...），其中
**Structure/Extraction** 是离 DocBench-meta 最近的"结构/全局属性"题——正是 DG 的主场
（检索够不到、程序能算），和 Kleister 的可检索跨度本质不同，值得一试。

本脚本：找到 PDFTriage 的题目文件（json/jsonl/csv，自动识别），对每题跑 dg_core.parse，
**按 category 分桶**报命中率，重点看 Structure/Extraction 够不够大、命中够不够高。

用法：python probe_pdftriage.py --src /root/autodl-tmp/pdftriage
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="pdftriage 克隆根目录")
ap.add_argument("--dg-core", default="/root/autodl-tmp/rag-L1/reproduce")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, a.dg_core)
try:
    import dg_core
    parse_fn = dg_core.parse
except Exception as e:
    print("!! dg_core.parse 不可用:", e)
    sys.exit(2)

QFIELDS = ("question", "query", "Question", "question_text", "text")  # PDFTriage/DocInstruct 用 text
TFIELDS = ("category", "question_type", "type", "q_type", "qtype", "label", "class")
# 过滤众包退化项（非问题）：N/A、空、太短
_JUNK = ("n/a", "na", "none", "no figure", "no table", "-")


def _field(d, names):
    for n in names:
        if isinstance(d, dict) and d.get(n) not in (None, "", []):
            v = d[n]
            return v[0] if isinstance(v, list) and v else v
    return None


def _load_records(src):
    """在克隆目录里找含 question 字段的 json/jsonl/csv，返回 [dict]。取命中最多的那个文件。"""
    best, best_path = [], None
    for path in glob.glob(os.path.join(src, "**", "*"), recursive=True):
        low = path.lower()
        if not low.endswith((".json", ".jsonl", ".csv")):
            continue
        recs = []
        try:
            if low.endswith(".csv"):
                with open(path, encoding="utf-8", errors="ignore") as f:
                    recs = list(csv.DictReader(f))
            elif low.endswith(".jsonl"):
                recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            else:
                obj = json.load(open(path, encoding="utf-8"))
                if isinstance(obj, dict):
                    obj = next((v for v in obj.values() if isinstance(v, list)), obj)
                recs = obj if isinstance(obj, list) else []
        except Exception:
            continue
        recs = [r for r in recs if isinstance(r, dict) and _field(r, QFIELDS)]
        if len(recs) > len(best):
            best, best_path = recs, path
    return best, best_path


recs, path = _load_records(a.src)
if not recs:
    print(f"!! 在 {a.src} 没找到含 question 字段的 json/jsonl/csv；把 `ls -R {a.src} | head -50` 贴回")
    sys.exit(2)
print(f"题目文件: {path}（{len(recs)} 题）")

by_cat_total, by_cat_hit = Counter(), Counter()
n = hit = 0
for r in recs:
    q = _field(r, QFIELDS)
    if not q or str(q).strip().lower() in _JUNK or len(str(q).strip()) < 8:
        continue                                   # 跳过 N/A 等众包退化项
    cat = str(_field(r, TFIELDS) or "?")
    fired = bool(parse_fn(str(q)))
    n += 1
    hit += fired
    by_cat_total[cat] += 1
    by_cat_hit[cat] += int(fired)
    if a.limit and n >= a.limit:
        break

print("\n================ PDFTriage 探针（零 API） ================")
print(f"取样 {n} 题；dg_core.parse 命中 {hit} 题（{hit / max(n, 1) * 100:.1f}%）")
print("---- 按 category 分桶（命中/总数）----")
for c in sorted(by_cat_total, key=lambda k: -by_cat_hit[k]):
    star = "  <- 目标类(结构/抽取)" if any(k in c.lower() for k in ("struct", "extract", "metadata")) else ""
    print(f"  {c:22s}: {by_cat_hit[c]:4d} / {by_cat_total[c]:4d}{star}")
print("判读：Structure/Extraction 命中够大 → 建索引跑 base vs +DG（这类题检索够不到、DG 能算，"
      "预期像 DocBench 一样大增益，且是已发表基准=可信的第二数据集）。")
