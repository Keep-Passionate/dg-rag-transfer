#!/usr/bin/env bash
# DocBench 对比基线一条龙：跑基线(meta 子集) -> 评测(复用主仓评测器) -> 对比。可续跑。
# 用法（跑前关学术加速）：bash run_baselines.sh longcontext|bm25|pal|all [N]
set -eo pipefail
MODE="${1:-all}"
N="${2:-0}"                                             # 0=全量；正整数=只前 N 篇

MAIN="${MAIN:-/root/autodl-tmp/rag-L1}"                 # 主仓（reproduce/评测器 + .env）
DOC="${DOC:-/root/autodl-tmp/DocBench_subset}"
PY="${PY:-/root/miniconda3/envs/rag/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EVOUT="$DOC/_eval_baselines"

# 与主实验同一 key/端点（公平对比：唯一变量是"怎么拿答案"）
export LLM_BINDING_API_KEY=$(grep -E '^LLM_BINDING_API_KEY=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
export LLM_BINDING_HOST=$(grep -E '^LLM_BINDING_HOST=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
[ -n "$LLM_BINDING_API_KEY" ] || { echo "!! 没从 $MAIN/.env 读到 LLM_BINDING_API_KEY"; exit 1; }

run_one() {
  echo "==== 跑基线 $1 (N=$N, meta-only) ===="
  $PY "$HERE/run_$1.py" --root "$DOC" --limit "$N"
}
case "$MODE" in
  longcontext|bm25|pal) run_one "$MODE" ;;
  all) run_one longcontext; run_one bm25; run_one pal ;;
  *) echo "用法: bash run_baselines.sh longcontext|bm25|pal|all [N]"; exit 1 ;;
esac

echo "==== 评测（增量，⚠️不可并发；judges 所有 qa_results_*.json）===="
$PY "$MAIN/reproduce/llm_answer_evaluator.py" --qa-data-dir "$DOC" -o "$EVOUT" \
    --api-key "$LLM_BINDING_API_KEY" --base-url "$LLM_BINDING_HOST" 2>&1 | tail -6

echo "==== 对比（meta 子集）===="
$PY "$HERE/compare_baselines.py" --root "$DOC" --eval "$EVOUT/llm_evaluation_results.json"
