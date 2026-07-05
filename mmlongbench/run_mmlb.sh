#!/usr/bin/env bash
# Run MMLongBench-Doc subsets with RAG-Anything.
#
# Default keeps the original experiment: base vs DG only.
# Set RUN_MM=1 to additionally run MM-only and DG+MM:
#   RUN_MM=1 bash mmlongbench/run_mmlb.sh smoke
#   RUN_MM=1 bash mmlongbench/run_mmlb.sh 25

set -eo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke) N=2 ;;
  25) N=25 ;;
  all) N=99999 ;;
  *) echo "usage: bash run_mmlb.sh smoke|25|all"; exit 1 ;;
esac

MAIN="${MAIN:-/root/autodl-tmp/rag-L1}"
MMLB="${MMLB:-/root/autodl-tmp/MMLB_subset}"
PARSE_OUT="${PARSE_OUT:-/root/autodl-tmp/mmlb_parse}"
WORK="${WORK:-/root/autodl-tmp/mmlb_storage}"
PY="${PY:-/root/miniconda3/envs/rag/bin/python}"
RUN_MM="${RUN_MM:-0}"
LOG="${LOG:-/root/autodl-tmp/mmlb_${MODE}.log}"

[ -f "$MMLB/mmlb_manifest.json" ] || {
  echo "!! Missing $MMLB/mmlb_manifest.json. Run prepare_mmlb.py first."
  exit 1
}
[ -d "$MAIN/reproduce" ] || { echo "!! MAIN does not look like the RAG-Anything repo: $MAIN"; exit 1; }

KEY=$(grep -E '^LLM_BINDING_API_KEY=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)
HOST=$(grep -E '^LLM_BINDING_HOST=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)
HOST="${HOST:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
[ -n "$KEY" ] || { echo "!! Could not read LLM_BINDING_API_KEY from $MAIN/.env"; exit 1; }

# Keep the paper-base condition clean. Individual runs override the switches
# they need, so old base/DG numbers remain comparable.
export ENABLE_VLM=false ENABLE_MODALITY_VLM=false ENABLE_AUTO_VLM=false ENABLE_EMR=false
export ENABLE_DOC_META=false ENABLE_NCG=false ENABLE_DOC_OUTLINE=false ENABLE_DOC_ANCHOR=false
export ENABLE_DOC_LOCATE=false ENABLE_RERANK=false ENABLE_RETRIEVAL_REFLECT=false SAVE_CONTEXT=false

index_ready() {
  local dir="$1"
  [ -f "$dir/kv_store_full_docs.json" ] || [ -f "$dir/vdb_chunks.json" ]
}

run_query() {
  local id="$1"
  local pdf="$2"
  local result="$3"
  shift 3
  if [ -f "$MMLB/$id/$result" ]; then
    echo "[$id] skip existing $result" | tee -a "$LOG"
    return 0
  fi
  echo "[$id] query -> $result" | tee -a "$LOG"
  env "$@" RESULT_NAME="$result" \
    "$PY" reproduce/query.py "$pdf" --working_dir "$WORK/$id" >>"$LOG" 2>&1 || true
}

cd "$MAIN"
ids=$("$PY" -c "import json; m=json.load(open('$MMLB/mmlb_manifest.json')); print('\n'.join(r['doc_id'] for r in m[:$N]))")

for id in $ids; do
  pdf=$(ls "$MMLB/$id"/*.pdf 2>/dev/null | head -1) || true
  [ -n "$pdf" ] || { echo "[$id] no PDF, skip" | tee -a "$LOG"; continue; }
  echo "=== $id $(basename "$pdf") $(date +%F-%H:%M) ===" | tee -a "$LOG"

  if ! index_ready "$WORK/$id"; then
    echo "[$id] indexing..." | tee -a "$LOG"
    "$PY" reproduce/index.py "$pdf" --working_dir "$WORK/$id" --output "$PARSE_OUT" >>"$LOG" 2>&1 || {
      echo "[$id] !! index failed, skip this document" | tee -a "$LOG"
      continue
    }
  fi

  run_query "$id" "$pdf" qa_results_mmlb_base.json \
    ENABLE_DG_CORE=false ENABLE_MM_GROUND=false ENABLE_MODALITY_VLM=false ENABLE_EMR=false

  run_query "$id" "$pdf" qa_results_mmlb_dg.json \
    ENABLE_DG_CORE=true ENABLE_MM_GROUND=false PARSE_OUTPUT_DIR="$PARSE_OUT" OUTPUT_DIR="$PARSE_OUT"

  if [ "$RUN_MM" = "1" ] || [ "$RUN_MM" = "true" ]; then
    run_query "$id" "$pdf" qa_results_mmlb_mm.json \
      ENABLE_DG_CORE=false ENABLE_MM_GROUND=true ENABLE_MODALITY_VLM=true ENABLE_EMR=true \
      PARSE_OUTPUT_DIR="$PARSE_OUT" OUTPUT_DIR="$PARSE_OUT" \
      MM_BROAD_VISUAL_FALLBACK="${MM_BROAD_VISUAL_FALLBACK:-true}" \
      MM_VISUAL_TOPK="${MM_VISUAL_TOPK:-3}" MM_TABLE_TOPK="${MM_TABLE_TOPK:-3}"

    run_query "$id" "$pdf" qa_results_mmlb_dg_mm.json \
      ENABLE_DG_CORE=true ENABLE_MM_GROUND=true ENABLE_MODALITY_VLM=true ENABLE_EMR=true \
      PARSE_OUTPUT_DIR="$PARSE_OUT" OUTPUT_DIR="$PARSE_OUT" \
      MM_BROAD_VISUAL_FALLBACK="${MM_BROAD_VISUAL_FALLBACK:-true}" \
      MM_VISUAL_TOPK="${MM_VISUAL_TOPK:-3}" MM_TABLE_TOPK="${MM_TABLE_TOPK:-3}"
  fi

  echo "[$id] done" | tee -a "$LOG"
done

echo "==== evaluation (do not run multiple evaluators concurrently) ====" | tee -a "$LOG"
"$PY" reproduce/llm_answer_evaluator.py --qa-data-dir "$MMLB" -o "$MMLB/_eval" \
  --api-key "$KEY" --base-url "$HOST" 2>&1 | tail -12 | tee -a "$LOG"

echo "==== comparison ====" | tee -a "$LOG"
"$PY" "$(dirname "$0")/compare_mmlb.py" --mmlb "$MMLB" 2>&1 | tee -a "$LOG"
