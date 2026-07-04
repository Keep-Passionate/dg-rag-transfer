#!/usr/bin/env bash
# Kleister-NDA 跨数据集迁移：RAG-Anything(主仓 index.py/query.py) 上 base vs +DG。
# 每篇：建图(index.py,贵)-> 查询 base(ENABLE_DG_CORE=false) / dg(=true) -> 评测 -> 对比。可续跑。
# 用法（跑前关学术加速）：bash run_kleister.sh smoke|25|all
# ⚠️ 先跑 prepare_kleister.py 探针；命中子集够大再建索引。NDA 每篇 ~6 页，比 MMLB 便宜得多。
set -eo pipefail
MODE="${1:-smoke}"
case "$MODE" in smoke) N=3 ;; 25) N=25 ;; all) N=99999 ;; *) echo "用法: bash run_kleister.sh smoke|25|all"; exit 1 ;; esac

MAIN="${MAIN:-/root/autodl-tmp/rag-L1}"                    # 主仓（含 reproduce/、.env）
KL="${KL:-/root/autodl-tmp/Kleister_subset}"               # prepare_kleister.py 落地目录
PARSE_OUT="${PARSE_OUT:-/root/autodl-tmp/kleister_parse}"  # index.py 解析产物；dg_core 从这找 content_list
WORK="${WORK:-/root/autodl-tmp/kleister_storage}"          # 每篇 RAG-Anything 索引
PY="${PY:-/root/miniconda3/envs/rag/bin/python}"
LOG="/root/autodl-tmp/kleister_${MODE}.log"

[ -f "$KL/kleister_manifest.json" ] || { echo "!! 先跑 prepare_kleister.py 生成 $KL/kleister_manifest.json"; exit 1; }
KEY=$(grep -E '^LLM_BINDING_API_KEY=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
HOST=$(grep -E '^LLM_BINDING_HOST=' "$MAIN/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
[ -n "$KEY" ] || { echo "!! 没从 $MAIN/.env 读到 LLM_BINDING_API_KEY"; exit 1; }

# 条件=主实验口径：base=paperbase(所有增强关)；dg=仅 ENABLE_DG_CORE。唯一变量是 DG。
export ENABLE_VLM=false ENABLE_MODALITY_VLM=false ENABLE_AUTO_VLM=false ENABLE_EMR=false \
  ENABLE_DOC_META=false ENABLE_NCG=false ENABLE_DOC_OUTLINE=false ENABLE_DOC_ANCHOR=false \
  ENABLE_DOC_LOCATE=false ENABLE_RERANK=false ENABLE_RETRIEVAL_REFLECT=false SAVE_CONTEXT=false

cd "$MAIN"
# 优先跑 parse 命中多的文档（manifest 已按命中降序）
ids=$($PY -c "import json;m=json.load(open('$KL/kleister_manifest.json'));print('\n'.join(r['doc_id'] for r in m[:$N]))")
for id in $ids; do
  doc=$(ls "$KL/$id"/*.pdf 2>/dev/null | head -1) || true
  [ -z "$doc" ] && doc=$(ls "$KL/$id"/*.txt 2>/dev/null | head -1) || true
  [ -z "$doc" ] && continue
  echo "=== $id $(basename "$doc") $(date +%H:%M) ===" | tee -a "$LOG"
  # 1) 建图（RAG-Anything；失败则跳过本篇）
  if [ ! -f "$WORK/$id/kv_store_full_docs.json" ]; then
    echo "[$id] 建图..." | tee -a "$LOG"
    $PY reproduce/index.py "$doc" --working_dir "$WORK/$id" --output "$PARSE_OUT" >>"$LOG" 2>&1 \
      || { echo "[$id] !! 建图失败，跳过" | tee -a "$LOG"; continue; }
  fi
  # 2) 查询 base（无 DG）
  if [ ! -f "$KL/$id/qa_results_kleister_base.json" ]; then
    ENABLE_DG_CORE=false RESULT_NAME=qa_results_kleister_base.json \
      $PY reproduce/query.py "$doc" --working_dir "$WORK/$id" >>"$LOG" 2>&1 || true
  fi
  # 3) 查询 dg（仅 DG）
  if [ ! -f "$KL/$id/qa_results_kleister_dg.json" ]; then
    ENABLE_DG_CORE=true PARSE_OUTPUT_DIR="$PARSE_OUT" RESULT_NAME=qa_results_kleister_dg.json \
      $PY reproduce/query.py "$doc" --working_dir "$WORK/$id" >>"$LOG" 2>&1 || true
  fi
  echo "[$id] 查询完成" | tee -a "$LOG"
done

echo "==== 评测（增量，⚠️不可并发）====" | tee -a "$LOG"
$PY reproduce/llm_answer_evaluator.py --qa-data-dir "$KL" -o "$KL/_eval" \
    --api-key "$KEY" --base-url "$HOST" 2>&1 | tail -6 | tee -a "$LOG"
echo "==== 对比 ====" | tee -a "$LOG"
$PY "$(dirname "$0")/compare_kleister.py" --kleister "$KL" 2>&1 | tee -a "$LOG"
