#!/usr/bin/env bash
# MMLongBench-Doc 跨数据集迁移：RAG-Anything(主仓 index.py/query.py) 上 base vs +DG。
# 每篇：建图(index.py,贵)-> 查询 base(ENABLE_DG_CORE=false) / dg(=true) -> 评测 -> 对比。可续跑。
# 用法（跑前关学术加速）：bash run_mmlb.sh smoke|25|all
# ⚠️ MMLB 均 47.5 页/篇，建图很贵：smoke≈20-40 分钟，25 篇需 nohup 数小时，全量 135 篇通宵级。
set -eo pipefail
MODE="${1:-smoke}"
case "$MODE" in smoke) N=2 ;; 25) N=25 ;; all) N=99999 ;; *) echo "用法: bash run_mmlb.sh smoke|25|all"; exit 1 ;; esac

MAIN="${MAIN:-/root/autodl-tmp/rag-L1}"                 # 主仓（含 reproduce/、.env）
MMLB="${MMLB:-/root/autodl-tmp/MMLB_subset}"
PARSE_OUT="${PARSE_OUT:-/root/autodl-tmp/mmlb_parse}"   # index.py 的解析产物；dg_core 从这找 content_list
WORK="${WORK:-/root/autodl-tmp/mmlb_storage}"           # 每篇 RAG-Anything 索引
PY="${PY:-/root/miniconda3/envs/rag/bin/python}"
LOG="/root/autodl-tmp/mmlb_${MODE}.log"

[ -f "$MMLB/mmlb_manifest.json" ] || { echo "!! 先跑 prepare_mmlb.py 生成 $MMLB/mmlb_manifest.json"; exit 1; }
KEY=$(grep -E '^LLM_BINDING_API_KEY=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
HOST=$(grep -E '^LLM_BINDING_HOST=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
[ -n "$KEY" ] || { echo "!! 没从 $MAIN/.env 读到 LLM_BINDING_API_KEY"; exit 1; }

# 条件=主实验口径：base=paperbase(所有增强关)；dg=仅 ENABLE_DG_CORE。其余开关全关，保证唯一变量是 DG。
export ENABLE_VLM=false ENABLE_MODALITY_VLM=false ENABLE_AUTO_VLM=false ENABLE_EMR=false \
  ENABLE_DOC_META=false ENABLE_NCG=false ENABLE_DOC_OUTLINE=false ENABLE_DOC_ANCHOR=false \
  ENABLE_DOC_LOCATE=false ENABLE_RERANK=false ENABLE_RETRIEVAL_REFLECT=false SAVE_CONTEXT=false

cd "$MAIN"
ids=$($PY -c "import json;m=json.load(open('$MMLB/mmlb_manifest.json'));print('\n'.join(r['doc_id'] for r in m[:$N]))")
for id in $ids; do
  pdf=$(ls "$MMLB/$id"/*.pdf 2>/dev/null | head -1) || true
  [ -z "$pdf" ] && continue
  echo "=== $id $(basename "$pdf") $(date +%H:%M) ===" | tee -a "$LOG"
  # 1) 建图（RAG-Anything，长文档 10-30 分钟；失败则跳过本篇）
  if [ ! -d "$WORK/$id/vdb_chunks.json" ] && [ ! -f "$WORK/$id/kv_store_full_docs.json" ]; then
    echo "[$id] 建图..." | tee -a "$LOG"
    $PY reproduce/index.py "$pdf" --working_dir "$WORK/$id" --output "$PARSE_OUT" >>"$LOG" 2>&1 \
      || { echo "[$id] !! 建图失败，跳过" | tee -a "$LOG"; continue; }
  fi
  # 2) 查询 base（无 DG）
  if [ ! -f "$MMLB/$id/qa_results_mmlb_base.json" ]; then
    ENABLE_DG_CORE=false RESULT_NAME=qa_results_mmlb_base.json \
      $PY reproduce/query.py "$pdf" --working_dir "$WORK/$id" >>"$LOG" 2>&1 || true
  fi
  # 3) 查询 dg（仅 DG）
  if [ ! -f "$MMLB/$id/qa_results_mmlb_dg.json" ]; then
    ENABLE_DG_CORE=true PARSE_OUTPUT_DIR="$PARSE_OUT" RESULT_NAME=qa_results_mmlb_dg.json \
      $PY reproduce/query.py "$pdf" --working_dir "$WORK/$id" >>"$LOG" 2>&1 || true
  fi
  echo "[$id] 查询完成" | tee -a "$LOG"
done

echo "==== 评测（增量，⚠️不可并发）====" | tee -a "$LOG"
$PY reproduce/llm_answer_evaluator.py --qa-data-dir "$MMLB" -o "$MMLB/_eval" \
    --api-key "$KEY" --base-url "$HOST" 2>&1 | tail -6 | tee -a "$LOG"
echo "==== 对比 ====" | tee -a "$LOG"
$PY "$(dirname "$0")/compare_mmlb.py" --mmlb "$MMLB" 2>&1 | tee -a "$LOG"
