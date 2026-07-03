# MMLongBench-Doc 跨数据集迁移（第二数据集）

固定骨干 RAG-Anything（主仓 `reproduce/index.py`+`query.py`），换到 MMLongBench-Doc 上跑 base vs +DG，
证 DG 的增益非过拟合 DocBench。**诚实 caveat**：MMLB 无独立 meta 题类、偏多模态取证，DG 可作用子集小，
主看"DG 触发子集 dg>base + 未触发零拖累"，而非 overall 大涨。

## 文件
- `prepare_mmlb.py` — MMLB → DocBench 式布局（`MMLB_subset/<mNNN>/{pdf, mNNN_qa.jsonl}`）+ 用 `dg_core.parse` 统计可作用子集（零 API）；产 `mmlb_manifest.json`（按 parse 命中降序）。
- `run_mmlb.sh smoke|25|all` — 建图→查询 base/dg→评测→对比，一条龙，可续跑。
- `compare_mmlb.py` — DG触发子集/未触发/overall 的 base vs dg + McNemar（零 LLM）。

## 口径
- base = paperbase（`ENABLE_DG_CORE=false`，其余增强开关全关）；dg = 仅 `ENABLE_DG_CORE=true`（+`PARSE_OUTPUT_DIR`）。唯一变量是 DG。
- 与主实验、GraphRAG/vanilla 迁移同一套模型/评测器；结果回填交接 13 表 `tab:dataset` 第二行 + 两版 .tex。

## 成本提醒
MMLB 均 47.5 页/篇，`index.py` 建 RAG-Anything 双图（LLM 抽取）**很贵**：smoke(2篇)≈20-40 分钟，25 篇数小时，全量 135 篇通宵级。先 smoke→看探针子集→25 篇→全量。
