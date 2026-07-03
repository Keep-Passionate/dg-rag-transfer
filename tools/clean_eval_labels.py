"""列出 / 精简 eval 结果里的 method 标签（非破坏：默认写新文件；--inplace 才覆盖且先备份 .bak）。

论文最终只保留少数几个 method（paperbase / ours / DG-RAG …），把历次消融的中间标签
（dgcore/dgcoreV2/dgcoreV2cov/dgAllfire…）从展示用的 eval 里剔除，方便出表。

用法：
  列出全部 method + 题数：  python clean_eval_labels.py --eval <llm_evaluation_results.json>
  精简保留：                python clean_eval_labels.py --eval <path> --keep paperbase,ours,dgAllfire --out <new.json>
  同时重命名：              ... --keep paperbase,ours,dgAllfire --rename dgAllfire=DG-RAG,ours=ours-orig
  就地覆盖(先自动备份.bak):  ... --keep ... --inplace
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True, help="llm_evaluation_results.json 路径")
ap.add_argument("--keep", default="", help="逗号分隔要保留的 method")
ap.add_argument("--rename", default="", help="逗号分隔 old=new")
ap.add_argument("--out", default="", help="输出文件（默认 <eval>_paper.json）")
ap.add_argument("--inplace", action="store_true", help="就地覆盖（先备份 .bak）")
a = ap.parse_args()

raw = json.load(open(a.eval, encoding="utf-8"))
wrapped = isinstance(raw, dict) and "results" in raw
recs = raw["results"] if wrapped else raw

counts = Counter(r.get("method") for r in recs)
print(f"共 {len(recs)} 条记录，{len(counts)} 个 method：")
for m, c in counts.most_common():
    print(f"  {str(m):32} {c}")

if not a.keep:
    print("\n（只列出。精简：加 --keep m1,m2,...  [--out 新文件]  [--rename old=new,...]）")
    raise SystemExit

keep = {s.strip() for s in a.keep.split(",") if s.strip()}
ren = dict(kv.split("=", 1) for kv in a.rename.split(",") if "=" in kv)
missing = keep - set(counts)
if missing:
    print(f"\n!! --keep 里这些 method 不存在（核对拼写）：{sorted(missing)}")
    raise SystemExit(2)

kept = [dict(r, method=ren.get(r.get("method"), r.get("method")))
        for r in recs if r.get("method") in keep]
out_obj = ({**raw, "results": kept} if wrapped else kept)

if a.inplace:
    shutil.copy(a.eval, a.eval + ".bak")
    out_path = a.eval
    print(f"\n已备份 -> {a.eval}.bak")
else:
    out_path = a.out or (str(Path(a.eval).with_suffix("")) + "_paper.json")
json.dump(out_obj, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"精简后 {len(kept)} 条（保留 {sorted(keep)}，重命名 {ren or '无'}）-> {out_path}")
