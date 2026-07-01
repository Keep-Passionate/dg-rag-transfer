#!/usr/bin/env bash
# 评测（rag 环境 python，⚠️不可并发，别同时开第二个）+ 对比 base vs dg。
# 用法：bash run_eval_compare.sh [top25.json|top50.json]
set -e
cd "$(dirname "$0")"
MAN="${1:-top25.json}"

: "${DG_CORE_DIR:?先 source env.sh}"; : "${GRAPHRAG_EVAL:?先 source env.sh}"

echo "==== 评测（增量，已判过的题自动跳过）===="
"$EVAL_PY" "$DG_CORE_DIR/llm_answer_evaluator.py" \
    --qa-data-dir "$GRAPHRAG_EVAL" \
    -o "$GRAPHRAG_EVAL/_eval" \
    --api-key "$DASHSCOPE_API_KEY" \
    --base-url "$DASHSCOPE_BASE"

echo ""
echo "==== 对比：GraphRAG base(原题) vs dg(增强题) · meta 子集 ===="
python compare_graphrag.py \
    --root "$DOCBENCH_ROOT" \
    --eval "$GRAPHRAG_EVAL/_eval/llm_evaluation_results.json" \
    --manifest "$MAN"
