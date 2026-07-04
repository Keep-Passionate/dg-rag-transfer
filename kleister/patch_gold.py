"""把金标补进【已跑出】的 qa_results（不重查询、不烧 LLM 钱）。

背景：早期 prepare 把 qa.jsonl 的 answer 写成空串，query.py 据此把 correct_answer 也写空，
评测器无金标可判。修复后的 prepare 已能生成带金标的 qa.jsonl；本脚本按 question 对齐，
把金标灌进同目录已有的 qa_results_kleister_*.json 的 correct_answer 字段。

用法：
  python prepare_kleister.py --src .../kleister-nda   # 先重生成带金标的 qa.jsonl
  python patch_gold.py                                # 再把金标补进已有结果
  rm -rf .../Kleister_subset/_eval                    # 清旧评测，强制重判
  bash run_kleister.sh all                            # 只会重评测+对比（建图/查询都跳过）
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--kleister", default="/root/autodl-tmp/Kleister_subset")
a = ap.parse_args()

norm = lambda q: " ".join((q or "").split())
n_files = n_fixed = 0
for folder in sorted(glob.glob(os.path.join(a.kleister, "k_*"))):
    qa = os.path.join(folder, os.path.basename(folder) + "_qa.jsonl")
    if not os.path.exists(qa):
        continue
    gold = {}
    with open(qa, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                gold[norm(d.get("question"))] = d.get("answer", "")
    for rf in glob.glob(os.path.join(folder, "qa_results_kleister_*.json")):
        recs = json.load(open(rf, encoding="utf-8"))
        changed = False
        for r in recs:
            k = norm(r.get("question"))
            if k in gold and gold[k] and not r.get("correct_answer"):
                r["correct_answer"] = gold[k]
                n_fixed += 1
                changed = True
        if changed:
            json.dump(recs, open(rf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        n_files += 1

print(f"补金标完成：处理 {n_files} 个结果文件，填补 {n_fixed} 条空金标。")
print("下一步：rm -rf", os.path.join(a.kleister, "_eval"), "然后 bash run_kleister.sh all（只重评测+对比）。")
