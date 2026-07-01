#!/usr/bin/env bash
# 全题型跑（meta + 非meta）→ 分题型 + overall + 非回归。索引已建则复用（很省）；
# 非meta题 DG 弃权→dg=base，不额外查询。用法：source env.sh 后  bash run_overall.sh [25|50]
set -eo pipefail
cd "$(dirname "$0")"
: "${GRAPHRAG_API_KEY:?先 export DASHSCOPE_API_KEY 并 source env.sh}"
N="${1:-25}"
if [ "$N" = "50" ]; then MAN=top50.json; else MAN=top25.json; fi
LOG="/root/autodl-tmp/graphrag_overall_${N}.log"

echo "==== 全题型跑 $N 篇（manifest=$MAN）——弹幕： tail -f $LOG ===="
python run_graphrag_docbench.py --manifest "$MAN" --method local --all-types 2>&1 | tee "$LOG"

echo ""
bash run_eval_compare.sh "$MAN" 2>&1 | tee -a "$LOG"
