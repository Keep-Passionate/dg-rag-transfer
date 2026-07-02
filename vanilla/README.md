# Vanilla / Native RAG 骨干（DG 迁移第三骨干）

最朴素的 RAG（Lewis 2020 形态）：**嵌入切块 → 稠密 top-k 检索 → qwen 生成**，无图、无 rerank、
无多模态、**无 LLM 建索引**（只嵌入，近乎免费）。与 GraphRAG 版共用 `dg_augmenter` / 主仓评测器 /
`../graphrag/compare_graphrag.py`，接口完全一致，只是 method 名为 `vanilla_base` / `vanilla_dg`。

## 文件
- `vanilla_rag.py` — 骨干核心：chunk_text / embed(批≤10) / build_index / retrieve / generate / answer。
- `run_vanilla_docbench.py` — 驱动：每篇建索引 + 逐题 base/dg，产出 `<VANILLA_EVAL>/<id>/qa_results_vanilla_{base,dg}.json`。
- `run_vanilla.sh` — 一条龙：索引→查询→评测→对比。

## 跑法（先 source 迁移仓的 env.sh）
```bash
cd /root/autodl-tmp/dg-rag-transfer/graphrag && source ./env.sh   # 设 key/DG_CORE_DIR/EVAL_PY/路径
cd ../vanilla
bash run_vanilla.sh smoke                # 2 篇打通
bash run_vanilla.sh 25 --all-types       # 25 篇全题型（分型+overall+非回归）
bash run_vanilla.sh all --all-types      # 全量
```
弹幕：`tail -f /root/autodl-tmp/vanilla_25.log`。索引产物在 `/root/autodl-tmp/vanilla_runs/`，结果在 `/root/autodl-tmp/vanilla_eval/`（与 GraphRAG 的目录隔离）。

**续跑**：已建索引/已出结果的篇自动跳过（meta 模式）。⚠️ `--all-types` 会重写全题型结果，故该模式**不跳过已查篇**——全量+overall 请一次 `nohup` 跑完，中断后重跑会重查烧钱（与 GraphRAG 版同行为）。敏感文档默认 `EXCLUDE_IDS=110,161,187,203,210` 跳过；vanilla 无 LLM 建索引审查风险，可试 `EXCLUDE_IDS=110 bash run_vanilla.sh 25` 纳入更多篇。

## 环境
用 `graphrag` conda 环境跑驱动（需 openai/numpy/pymupdf/tiktoken；切块 ~600 token/100 重叠走 tiktoken，缺则按字符回退）；评测器用 `EVAL_PY`（rag 环境，依赖 lightrag）。
key 只从 `DASHSCOPE_API_KEY` 读。百炼 embedding 单批 ≤10（已在 `embed()` 内分批）。

## 预期
base 在 meta 上应比 GraphRAG 高一些（切块能捞到含作者/页码的原文块），DG 再抬；非 meta 各型 Δ≈0（DG 弃权→恒等回退）。
结果回填 `D:\project\交接\13_迁移实验_论文表格与文字.md` 表 A.1 的 "Vanilla dense RAG" 行。
