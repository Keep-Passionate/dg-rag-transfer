#!/usr/bin/env bash
# 迁移实验环境变量。用法：先 export DASHSCOPE_API_KEY=<你的百炼key>，再 `source env.sh`。
# 只从 env 读密钥，绝不写进代码/仓库。

: "${DASHSCOPE_API_KEY:?请先 export DASHSCOPE_API_KEY=<你的百炼key> 再 source env.sh}"
export GRAPHRAG_API_KEY="$DASHSCOPE_API_KEY"                    # GraphRAG settings.yaml 用 ${GRAPHRAG_API_KEY}
export DASHSCOPE_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

# ★关键★ GraphRAG 3.x 用 litellm；litellm 启动会去 GitHub 拉模型价目表，国内关了学术加速会无限卡。
# 强制用本地自带价目表、不联网。缺了这个，graphrag init/index/query 全都会卡死。
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# tiktoken 首次会联网下编码文件（openaipublic.blob，被墙）。固定缓存目录；请在学术加速开着时预下载一次：
#   source /etc/network_turbo && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/root/autodl-tmp/.tiktoken_cache}"
mkdir -p "$TIKTOKEN_CACHE_DIR"

# dg_core.py + llm_answer_evaluator.py 所在（接地层 + 评测器）。
# 服务器实测最终版在 rag-L1/reproduce（分支 feat/dg-core-coverage，dg_core.py=44008B）；
# 注意 RAG-Anything-thesis 在 main 分支、没有 reproduce/dg_core（也别用它那份旧评测器）。
# 自动纠正：外部若已 export 了错的 DG_CORE_DIR（指到没有 dg_core.py 的目录），这里强制改回。
if [ -z "${DG_CORE_DIR:-}" ] || [ ! -f "${DG_CORE_DIR}/dg_core.py" ]; then
  export DG_CORE_DIR=/root/autodl-tmp/rag-L1/reproduce
fi
# 评测器依赖 lightrag/pandas，须用主实验 rag 环境的 python。
export EVAL_PY="${EVAL_PY:-/root/miniconda3/envs/rag/bin/python}"

export PARSE_OUTPUT_DIR="${PARSE_OUTPUT_DIR:-/root/autodl-tmp/content_lists}"
export DOCBENCH_ROOT="${DOCBENCH_ROOT:-/root/autodl-tmp/DocBench_subset}"
export GRAPHRAG_RUNS="${GRAPHRAG_RUNS:-/root/autodl-tmp/graphrag_runs}"     # 每篇索引产物
export GRAPHRAG_EVAL="${GRAPHRAG_EVAL:-/root/autodl-tmp/graphrag_eval}"     # 我们的结果文件（与主实验结果隔离）

echo "DG_CORE_DIR       = $DG_CORE_DIR"
echo "EVAL_PY           = $EVAL_PY"
echo "PARSE_OUTPUT_DIR  = $PARSE_OUTPUT_DIR"
echo "DOCBENCH_ROOT     = $DOCBENCH_ROOT"
echo "GRAPHRAG_RUNS     = $GRAPHRAG_RUNS"
echo "GRAPHRAG_EVAL     = $GRAPHRAG_EVAL"
[ -f "$DG_CORE_DIR/dg_core.py" ] && echo "OK  dg_core 找到" || echo "!! dg_core 没找到，请手动 export DG_CORE_DIR=<主仓>/reproduce"
[ -x "$EVAL_PY" ] && echo "OK  rag 环境 python 存在" || echo "!! $EVAL_PY 不存在，请改 EVAL_PY 指向装了 lightrag 的环境"
