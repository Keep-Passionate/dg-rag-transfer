"""DUDE 小样本探针：dg_core.parse 可作用子集统计（零 API、只下标注 JSON、不碰图片）。

新版不再用 HF 脚本加载器（新版 datasets 已移除脚本型数据集）。改为直接下载 DUDE 官方
公开测试集标注 JSON（Zenodo，几 MB，纯问题+答案类型，无图片），对每题跑 dg_core.parse，
按 answer_type 分桶报命中率，判断有无值得建索引的可作用子集。

DUDE 问题本身就是自然语言，直接 parse 即可，无需文档。

用法：
  python probe_dude.py --limit 400
  python probe_dude.py --url <别的DUDE_gt.json> --limit 400
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import Counter

# 你在服务器 DUDE_loader.py 里找到的官方公开测试集金标（Zenodo，无图片）
DEFAULT_URL = ("https://zenodo.org/record/7763635/files/"
               "2023-03-23_DUDE_gt_test_PUBLIC.json?download=1")

ap = argparse.ArgumentParser()
ap.add_argument("--dg-core", default="/root/autodl-tmp/rag-L1/reproduce")
ap.add_argument("--url", default=DEFAULT_URL)
ap.add_argument("--cache", default="/root/autodl-tmp/dude_gt.json")
ap.add_argument("--limit", type=int, default=0, help="0=全部；>0 只看前 N 题")
a = ap.parse_args()

sys.path.insert(0, a.dg_core)
try:
    import dg_core
    parse_fn = dg_core.parse
except Exception as e:
    print("!! dg_core.parse 不可用:", e)
    sys.exit(2)

if not os.path.exists(a.cache):
    print(f"下载 DUDE 标注 -> {a.cache} …（走学术加速）")
    urllib.request.urlretrieve(a.url, a.cache)
raw = json.load(open(a.cache, encoding="utf-8"))

# 结构容错：可能是 list，或 {'data'/'annotations'/'questions': [...]}
items = raw
if isinstance(raw, dict):
    for k in ("data", "annotations", "questions", "dataset"):
        if isinstance(raw.get(k), list):
            items = raw[k]
            break
if not isinstance(items, list):
    print("!! 没解析出题目列表，键有：", list(raw)[:10] if isinstance(raw, dict) else type(raw))
    sys.exit(2)


def field(d, names):
    for n in names:
        v = d.get(n)
        if v not in (None, "", []):
            return v[0] if isinstance(v, list) and v else v
    return None


by_type_total = Counter()
by_type_hit = Counter()
n = hit = 0
for ex in items:
    if not isinstance(ex, dict):
        continue
    q = field(ex, ("question", "query", "Question"))
    if not q:
        continue
    t = field(ex, ("answer_type", "answers_type", "qtype", "type", "category")) or "?"
    fired = bool(parse_fn(str(q)))
    n += 1
    hit += fired
    by_type_total[str(t)] += 1
    by_type_hit[str(t)] += int(fired)
    if a.limit and n >= a.limit:
        break

print("\n================ DUDE 小样本探针（零 API） ================")
print(f"取样 {n} 题；dg_core.parse 命中 {hit} 题（{hit / max(n, 1) * 100:.1f}%）")
print("---- 按答案类型分桶（命中/总数）----")
for t in sorted(by_type_total, key=lambda k: -by_type_hit[k]):
    print(f"  {t:20s}: {by_type_hit[t]:4d} / {by_type_total[t]:4d}")
print("判读：命中子集占比够大且集中在 extractive/counting/date 才值得建索引跑 ±DG；"
      "若像 MMLB/Kleister 一样近零有效命中 → 记 Limitation，不建索引。")
