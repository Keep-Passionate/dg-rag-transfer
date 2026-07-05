#!/usr/bin/env bash
# Run DocBench subsets with the portable multimodal grounding layer on
# RAG-Anything indexes. Results are written into the original DocBench document
# directories; evaluation is restricted through a symlinked subset root.
#
# Usage:
#   bash docbench/run_raganything_mm.sh smoke
#   bash docbench/run_raganything_mm.sh 10
#   bash docbench/run_raganything_mm.sh 25

set -eo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke) N=2 ;;
  10) N=10 ;;
  25) N=25 ;;
  50) N=50 ;;
  all) N=999999 ;;
  *) echo "usage: bash run_raganything_mm.sh smoke|10|25|50|all"; exit 1 ;;
esac

MAIN="${MAIN:-/root/autodl-tmp/rag-L1}"
DOCBENCH="${DOCBENCH:-/root/autodl-tmp/DocBench_subset}"
MANIFEST="${MANIFEST:-/root/autodl-tmp/dg-rag-transfer/graphrag/top25.json}"
WORK="${WORK:-/root/autodl-tmp/RAG-Anything-thesis/rag_storage_baseline}"
PARSE_OUT="${PARSE_OUT:-/root/autodl-tmp/RAG-Anything-thesis/output}"
PY="${PY:-/root/miniconda3/envs/rag/bin/python}"
SUBSET_ROOT="${SUBSET_ROOT:-/root/autodl-tmp/docbench_mm_${MODE}_subset}"
EVAL_OUT="${EVAL_OUT:-/root/autodl-tmp/docbench_mm_${MODE}_eval}"
LOG="${LOG:-/root/autodl-tmp/docbench_mm_${MODE}.log}"
RUN_BASE="${RUN_BASE:-0}"

[ -d "$MAIN/reproduce" ] || { echo "!! MAIN does not look like RAG-Anything: $MAIN"; exit 1; }
[ -d "$DOCBENCH" ] || { echo "!! DOCBENCH not found: $DOCBENCH"; exit 1; }

KEY=$(grep -E '^LLM_BINDING_API_KEY=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)
HOST=$(grep -E '^LLM_BINDING_HOST=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)
HOST="${HOST:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
[ -n "$KEY" ] || { echo "!! Could not read LLM_BINDING_API_KEY from $MAIN/.env"; exit 1; }

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
  if [ -f "$DOCBENCH/$id/$result" ]; then
    echo "[$id] skip existing $result" | tee -a "$LOG"
    return 0
  fi
  echo "[$id] query -> $result" | tee -a "$LOG"
  env "$@" RESULT_NAME="$result" \
    "$PY" reproduce/query.py "$pdf" --working_dir "$WORK/$id" >>"$LOG" 2>&1 || true
}

ids_from_manifest() {
  "$PY" - "$MANIFEST" "$N" <<'PY'
import json, sys
path, limit = sys.argv[1], int(sys.argv[2])
data = json.load(open(path, encoding="utf-8"))
rows = data.get("docs", data) if isinstance(data, dict) else data
for row in rows[:limit]:
    print(row.get("doc_id", row.get("id")) if isinstance(row, dict) else row)
PY
}

ids_from_docbench() {
  "$PY" - "$DOCBENCH" "$N" <<'PY'
import os, sys
root, limit = sys.argv[1], int(sys.argv[2])
ids = [d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, f"{d}_qa.jsonl"))]
ids.sort(key=lambda x: int(x) if x.isdigit() else x)
for doc_id in ids[:limit]:
    print(doc_id)
PY
}

mkdir -p "$SUBSET_ROOT"
cd "$MAIN"

if [ -f "$MANIFEST" ]; then
  ids=$(ids_from_manifest)
else
  ids=$(ids_from_docbench)
fi

for id in $ids; do
  [ -d "$DOCBENCH/$id" ] || { echo "[$id] missing doc dir, skip" | tee -a "$LOG"; continue; }
  ln -sfn "$DOCBENCH/$id" "$SUBSET_ROOT/$id"
  pdf=$(find "$DOCBENCH/$id" -maxdepth 1 -iname '*.pdf' | head -1)
  [ -n "$pdf" ] || { echo "[$id] no PDF, skip" | tee -a "$LOG"; continue; }
  echo "=== $id $(basename "$pdf") $(date +%F-%H:%M) ===" | tee -a "$LOG"

  if ! index_ready "$WORK/$id"; then
    echo "[$id] indexing..." | tee -a "$LOG"
    "$PY" reproduce/index.py "$pdf" --working_dir "$WORK/$id" --output "$PARSE_OUT" >>"$LOG" 2>&1 || {
      echo "[$id] !! index failed, skip" | tee -a "$LOG"
      continue
    }
  fi

  if [ "$RUN_BASE" = "1" ] || [ "$RUN_BASE" = "true" ]; then
    run_query "$id" "$pdf" qa_results_mm_base_docbench.json \
      ENABLE_DG_CORE=false ENABLE_MM_GROUND=false ENABLE_MODALITY_VLM=false ENABLE_EMR=false
  fi

  run_query "$id" "$pdf" qa_results_mmground_docbench.json \
    ENABLE_DG_CORE=false ENABLE_MM_GROUND=true ENABLE_MODALITY_VLM=true ENABLE_EMR=true \
    MM_ENABLE_VISUAL_ROUTING="${MM_ENABLE_VISUAL_ROUTING:-false}" \
    MM_ALLOW_RANKED_VISUAL="${MM_ALLOW_RANKED_VISUAL:-false}" \
    MM_BROAD_VISUAL_FALLBACK="${MM_BROAD_VISUAL_FALLBACK:-false}" \
    MM_FORCE_VLM="${MM_FORCE_VLM:-false}" \
    MM_VISUAL_TOPK="${MM_VISUAL_TOPK:-2}" MM_TABLE_TOPK="${MM_TABLE_TOPK:-2}" \
    PARSE_OUTPUT_DIR="$PARSE_OUT" OUTPUT_DIR="$PARSE_OUT"

  run_query "$id" "$pdf" qa_results_dg_mm_docbench.json \
    ENABLE_DG_CORE=true ENABLE_MM_GROUND=true ENABLE_MODALITY_VLM=true ENABLE_EMR=true \
    MM_ENABLE_VISUAL_ROUTING="${MM_ENABLE_VISUAL_ROUTING:-false}" \
    MM_ALLOW_RANKED_VISUAL="${MM_ALLOW_RANKED_VISUAL:-false}" \
    MM_BROAD_VISUAL_FALLBACK="${MM_BROAD_VISUAL_FALLBACK:-false}" \
    MM_FORCE_VLM="${MM_FORCE_VLM:-false}" \
    MM_VISUAL_TOPK="${MM_VISUAL_TOPK:-2}" MM_TABLE_TOPK="${MM_TABLE_TOPK:-2}" \
    PARSE_OUTPUT_DIR="$PARSE_OUT" OUTPUT_DIR="$PARSE_OUT"
done

echo "==== evaluation on subset root: $SUBSET_ROOT ====" | tee -a "$LOG"
"$PY" reproduce/llm_answer_evaluator.py --qa-data-dir "$SUBSET_ROOT" -o "$EVAL_OUT" \
  --api-key "$KEY" --base-url "$HOST" 2>&1 | tail -12 | tee -a "$LOG"

echo "==== comparison ====" | tee -a "$LOG"
"$PY" /root/autodl-tmp/dg-rag-transfer/docbench/compare_docbench_mm.py \
  --docbench "$SUBSET_ROOT" --eval "$EVAL_OUT/llm_evaluation_results.json" 2>&1 | tee -a "$LOG"
