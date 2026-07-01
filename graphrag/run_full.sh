#!/usr/bin/env bash
# 全量 · 全题型 · 断点续跑。DocBench_subset 下全部文档；索引/结果已存在就跳过——
# 半夜断线/关机后，重新跑一次本脚本即可接着往下走，不重复烧钱。
# 用法：source env.sh 后   nohup bash run_full.sh >/dev/null 2>&1 &   （弹幕 tail -f 下面的 LOG）
set -eo pipefail
cd "$(dirname "$0")"
: "${GRAPHRAG_API_KEY:?先 export DASHSCOPE_API_KEY 并 source env.sh}"
LOG="/root/autodl-tmp/graphrag_full.log"
# 已知内容审查必失败的敏感文档（含 manifest 的 exclude_ids），跳过省时；发现新的可追加。
EXCL="${EXCLUDE_IDS:-110,161,187,203,210}"

echo "==== 全量·全题型 断点续跑（exclude=$EXCL）$(date) —— tail -f $LOG ====" | tee -a "$LOG"
python run_graphrag_docbench.py --all-docs --all-types --method local \
    --exclude "$EXCL" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
# 评测(增量,判过的跳过) + 全量分型/overall 对比
bash run_eval_compare.sh --all-docs 2>&1 | tee -a "$LOG"
