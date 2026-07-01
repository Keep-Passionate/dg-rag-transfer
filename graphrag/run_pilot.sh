#!/usr/bin/env bash
# 试点：跑 N 篇（25 或 50）→ 评测 → 对比。可续跑（已建索引/已出结果的篇自动跳过）。
# 用法：source env.sh 后  bash run_pilot.sh 25   （或 50）
set -e
cd "$(dirname "$0")"
: "${GRAPHRAG_API_KEY:?先 export DASHSCOPE_API_KEY 并 source env.sh}"
N="${1:-25}"
if [ "$N" = "50" ]; then MAN=top50.json; else MAN=top25.json; fi
LOG="/root/autodl-tmp/graphrag_run_${N}.log"

echo "==== 跑 $N 篇（manifest=$MAN）——弹幕另开终端： tail -f $LOG ===="
python run_graphrag_docbench.py --manifest "$MAN" --method local 2>&1 | tee "$LOG"

echo ""
bash run_eval_compare.sh "$MAN"
