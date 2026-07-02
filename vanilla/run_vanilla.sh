#!/usr/bin/env bash
# Vanilla/native RAG：建索引(嵌入,近乎免费)+查询+评测+对比，一条龙。可续跑。
# 前置：先 `source ../graphrag/env.sh`（设 DASHSCOPE_API_KEY / DG_CORE_DIR / EVAL_PY / DOCBENCH_ROOT / PARSE_OUTPUT_DIR）。
# 用法：bash run_vanilla.sh [smoke|25|50|all] [--all-types]
set -eo pipefail
cd "$(dirname "$0")"
: "${DASHSCOPE_API_KEY:?先 source ../graphrag/env.sh}"
export DASHSCOPE_BASE="${DASHSCOPE_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
# vanilla 骨干在线走论文口径的门控 A（自检阈值）；只在本脚本进程内生效，不影响 GraphRAG 的 all-fire 跑
export DG_GATE="${DG_GATE:-self_check}"
echo "DG_GATE=$DG_GATE（self_check=在线门控A）"
export VANILLA_EVAL="${VANILLA_EVAL:-/root/autodl-tmp/vanilla_eval}"
export DOCBENCH_ROOT="${DOCBENCH_ROOT:-/root/autodl-tmp/DocBench_subset}"
EVAL_PY="${EVAL_PY:-/root/miniconda3/envs/rag/bin/python}"
DGD="${DG_CORE_DIR:-/root/autodl-tmp/rag-L1/reproduce}"

MODE="${1:-25}"; shift || true
EXTRA="$*"                    # 例如 --all-types
case "$MODE" in
  smoke) SEL="--limit 2 --manifest ../graphrag/top25.json"; SCOPE="--manifest ../graphrag/top25.json" ;;
  25)    SEL="--manifest ../graphrag/top25.json";           SCOPE="--manifest ../graphrag/top25.json" ;;
  50)    SEL="--manifest ../graphrag/top50.json";           SCOPE="--manifest ../graphrag/top50.json" ;;
  all)   SEL="--all-docs";                                  SCOPE="--all-docs" ;;
  *)     SEL="--docs $MODE --manifest ../graphrag/top25.json"; SCOPE="--manifest ../graphrag/top25.json" ;;
esac
LOG="/root/autodl-tmp/vanilla_${MODE}.log"

echo "==== Vanilla $MODE $EXTRA —— tail -f $LOG ====" | tee "$LOG"
python run_vanilla_docbench.py $SEL --exclude "${EXCLUDE_IDS:-110,161,187,203,210}" $EXTRA 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== 评测（rag 环境 python，⚠️不可并发）====" | tee -a "$LOG"
"$EVAL_PY" "$DGD/llm_answer_evaluator.py" \
    --qa-data-dir "$VANILLA_EVAL" -o "$VANILLA_EVAL/_eval" \
    --api-key "$DASHSCOPE_API_KEY" --base-url "$DASHSCOPE_BASE" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== 对比：Vanilla base vs dg（分题型+overall）====" | tee -a "$LOG"
python ../graphrag/compare_graphrag.py \
    --root "$DOCBENCH_ROOT" \
    --eval "$VANILLA_EVAL/_eval/llm_evaluation_results.json" \
    $SCOPE --base-method vanilla_base --dg-method vanilla_dg 2>&1 | tee -a "$LOG"
