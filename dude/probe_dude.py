"""DUDE（ICCV'23）小样本探针：dg_core.parse 可作用子集统计（零 API 成本）。

DUDE 是多页文档 VQA，答案类型含 extractive / abstractive / list / **not-answerable**，
结构最像 DocBench 但视觉稀释、无独立 metadata 类（=MMLB 同款风险）→ **必须先探针**（交接 24 §4）。
DUDE 问题本身就是自然语言，直接 `dg_core.parse(question)` 即可，无需文档、不花钱。

本脚本：小样本加载 DUDE → 每题跑 parse → 按 answer_type 分桶统计命中，判断有无值得建索引的可作用子集。

用法：
  python probe_dude.py --limit 300                 # HF 流式取前 300 题（省下载）
  python probe_dude.py --limit 300 --split val
"""
import argparse
import sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--dg-core", default="/root/autodl-tmp/rag-L1/reproduce")
ap.add_argument("--split", default="val", help="val | test（test 通常无金标 answer_type）")
ap.add_argument("--limit", type=int, default=300, help="小样本题数上限")
ap.add_argument("--config", default="Amazon_original")
a = ap.parse_args()

sys.path.insert(0, a.dg_core)
try:
    import dg_core
    parse_fn = dg_core.parse
except Exception as e:
    print("!! dg_core.parse 不可用:", e)
    sys.exit(2)

try:
    from datasets import load_dataset
except Exception as e:
    print("!! 需要 `pip install datasets`:", e)
    sys.exit(2)


def _iter_examples():
    """优先流式（不下整包）；失败再整载切片。"""
    try:
        ds = load_dataset("jordyvl/DUDE_loader", a.config, split=a.split, streaming=True,
                          trust_remote_code=True)
        for i, ex in enumerate(ds):
            if i >= a.limit:
                break
            yield ex
        return
    except Exception as e:
        print(f"（流式失败，回退整载：{e}）")
    ds = load_dataset("jordyvl/DUDE_loader", a.config, split=a.split, trust_remote_code=True)
    for i in range(min(a.limit, len(ds))):
        yield ds[i]


def _qfield(ex):
    for k in ("question", "questions", "query"):
        if ex.get(k):
            v = ex[k]
            return v[0] if isinstance(v, list) and v else str(v)
    return ""


def _tfield(ex):
    for k in ("answer_type", "answers_type", "type", "category"):
        if ex.get(k):
            v = ex[k]
            return v[0] if isinstance(v, list) and v else str(v)
    return "?"


by_type_total = Counter()
by_type_hit = Counter()
n = hit = 0
for ex in _iter_examples():
    q = _qfield(ex)
    if not q:
        continue
    t = _tfield(ex)
    fired = bool(parse_fn(q))
    n += 1
    hit += fired
    by_type_total[t] += 1
    by_type_hit[t] += int(fired)

print("\n================ DUDE 小样本探针（零 API） ================")
print(f"split={a.split}  取样 {n} 题；dg_core.parse 命中 {hit} 题（{hit / max(n, 1) * 100:.1f}%）")
print("---- 按答案类型分桶（命中/总数）----")
for t in sorted(by_type_total, key=lambda k: -by_type_hit[k]):
    print(f"  {t:20s}: {by_type_hit[t]:4d} / {by_type_total[t]:4d}")
print("判读：命中子集占比够大（且集中在 extractive/counting/date 类）才值得建索引跑 ±DG；"
      "若像 MMLB 一样近零命中 → 记 Limitation，不建索引。")
