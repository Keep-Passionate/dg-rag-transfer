#!/usr/bin/env bash
# SMOKE：top25 前 2 篇，打通 建输入→索引→查询→评测→对比 全链（花几毛钱验证接线）。
# 用法：source env.sh 后  bash run_smoke.sh
set -eo pipefail          # 驱动失败即停，不再往下刷评测/对比的连带报错
cd "$(dirname "$0")"
: "${GRAPHRAG_API_KEY:?先 export DASHSCOPE_API_KEY 并 source env.sh}"

echo "==== SMOKE：top25 前 2 篇（doc 0, 102）===="
python run_graphrag_docbench.py --limit 2 --method local 2>&1 | tee /root/autodl-tmp/graphrag_smoke.log

echo ""
bash run_eval_compare.sh top25.json
echo ""
echo "SMOKE 完成。若 base/dg 都出了数字、且驱动没报配置错，就可以 bash run_pilot.sh 25 上 25 篇。"
