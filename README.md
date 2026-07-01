# DG-RAG Transfer Experiments

证明 DG-RAG 的**确定性接地层（4 算子代数 + 自检门控 A，零 LLM）骨干无关**：把同一个
`dg_augmenter.augment()` 接到**不同 RAG 骨干**前面，看**元数据(meta)子集**的增益能否复现
（非 meta 按构造不变）。这是把主实验（RAG-Anything 上 meta 43.4%→66.3%，+22.9pp，p<0.001）
的贡献从"修某一个系统"抬升为"任何切块/图 RAG 都受益的通用前置层"。

## 接法（核心，骨干本身不改）
对每道题：`aug_q, meta = augment(question, pdf_path)` → 把 `aug_q` 交给骨干的 query。
门控弃权时 `aug_q == question`（**恒等回退**，骨干永不受损）。结果记录里存**原题**（对金标）。

## 仓库结构
```
dg_augmenter.py            骨干无关问题增强器（包 dg_core.ground）。DG_CORE_DIR 指向 dg_core 目录
graphrag/
  build_input.py           MinerU content_list → GraphRAG 输入 .txt（两骨干吃同一份解析，公平+省钱）
  patch_settings.py        把 graphrag init 生成的 settings.yaml 改成 qwen/text-embedding-v3/dashscope（自适应版本）
  run_graphrag_docbench.py 驱动：每篇 init→patch→index，逐题查询 ±augment，产出 qa_results_graphrag_*.json
  compare_graphrag.py      base(原题) vs dg(增强题) 准确率 + 配对 McNemar（读 eval JSON，零 LLM）
  top25.json / top50.json  固定 meta 子集（top-25=53 题 / top-50=103 题）
  env.sh                   环境变量（DG_CORE_DIR 自动探测 / 路径 / EVAL_PY=rag 环境 python）
  run_smoke.sh             2 篇 smoke：打通全链
  run_pilot.sh N           跑 N 篇（25/50）+ 评测 + 对比
  run_eval_compare.sh      评测（rag 环境 python，⚠️不可并发）+ 对比
```
> ⚠️ **两个 conda 环境**：驱动/索引/查询用新建的 `graphrag` 环境；评测器依赖 lightrag/pandas，
> 用主实验的 `rag` 环境 python（`env.sh` 里 `EVAL_PY`，默认 `/root/miniconda3/envs/rag/bin/python`）。
> GraphRAG（微软）本体是 **pip 包**，装进服务器上一个**独立 conda 环境**（不碰跑主实验的 `rag` 环境）；
> 它的每篇索引产物落在服务器 `/root/autodl-tmp/graphrag_runs/`，不入库。本仓只放**胶水/配置/对比**代码。

## 复用主仓的评测链（已核对，无需改评测器）
1. 驱动为每篇产出两个结果文件 `<docdir>/qa_results_graphrag_base.json`、`..._graphrag_dg.json`，
   格式 `[{"question":原题, "answer":输出, "correct_answer":金标}]`。
2. `python reproduce/llm_answer_evaluator.py --qa-data-dir <graphrag子集根>`：
   glob `**/qa_results_*.json`，doc_id=同目录 PDF 名，method=文件名去 `qa_results_` 前缀，
   判分写 `llm_evaluation_results.json`（accuracy 0/1）。EVAL_MODEL 默认 qwen-plus。**⚠️ 不可并发**。
3. `python graphrag/compare_graphrag.py --root <DocBench_subset> --eval <llm_evaluation_results.json>`。

## 模型/端点（照抄主仓 reproduce/common.py 的 env）
- chat：`qwen-plus`  ·  embedding：`text-embedding-v3`(dim 1024)  ·  vision（本迁移不用）：`qwen-vl-max`
- OpenAI 兼容端点：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- `api_key` = 环境变量 `DASHSCOPE_API_KEY`（**只从 env 读，绝不硬编码/入库**）

## 计划（严格控成本，先小后大）
1. **GraphRAG · 先 2–3 篇 smoke** 打通 索引→查询→评测→对比 全链 → **再 25 篇试点** → 信号好再 50 篇。
2. 一个 vanilla 稠密切块 RAG（无图、无 LLM 索引，近乎免费）：证明"任何切块 RAG 都受益"。
3. RAG-Anything 换一个测试集：证明换数据集仍成立（非过拟合 DocBench）。

## 主结果（对照，已定稿）
- meta 258：paperbase 43.4% → DG-RAG 66.3%（+22.9pp；test 43.8%→68.5%，McNemar χ²≈28.3, p<0.001）。
- overall 1102：72.6% → 78.0%（+5.4pp，全部来自 meta）。GraphRAG 上的"成功"= 也出现量级相近的
  **meta 正增益、非 meta 基本不变**。

## 服务器运行流程（一条龙）
```bash
# ① 学术加速仅用于 git/pip（跑实验前会关掉，因为查询走阿里云百炼）
source /etc/network_turbo
cd /root/autodl-tmp && git clone https://github.com/Keep-Passionate/dg-rag-transfer.git \
  || (cd dg-rag-transfer && git pull)
conda create -y -n graphrag python=3.11 && conda activate graphrag
pip install graphrag pymupdf pyyaml pandas
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # ② 关学术加速

# ③ 配环境（key 只从 env 读）
cd /root/autodl-tmp/dg-rag-transfer/graphrag
export DASHSCOPE_API_KEY=<你的百炼key>
source ./env.sh

# ④ 先 smoke（2 篇，几毛钱验证接线）
bash run_smoke.sh
# ⑤ smoke 没问题 → 25 篇 → 看结果 → 50 篇
bash run_pilot.sh 25
bash run_pilot.sh 50
```
另开终端看弹幕：`tail -f /root/autodl-tmp/graphrag_smoke.log`（或 `graphrag_run_25.log`）。

## 当前状态（2026-07-01）
- ✅ **SMOKE 打通**（GraphRAG 3.1.0，2 篇 6 题）：`graphrag_base 0/6=0.0%` → `graphrag_dg 5/6=83.3%`，
  配对 dg 净增 +5（独对5/独错0，McNemar p=0.0625，样本小不显著但方向对）。**DG meta 增益在 GraphRAG 骨干复现**。
- ✅ 全链验证：build_input → `graphrag index`(qwen) → `graphrag query` base/dg → 评测器判分 → compare。
- ⏳ 正在跑 `run_pilot.sh 25`（23 篇需新建索引，~1–2h；大/敏感文档可能内容审查失败被跳过）→ 出稳定数字+显著性。

### GraphRAG 3.1.0 踩坑备忘（都已在代码/脚本中处理）
1. litellm 启动联网拉价目表 → 国内卡死：`LITELLM_LOCAL_MODEL_COST_MAP=True`。
2. tiktoken 下编码文件被墙：预缓存 + 固定 `TIKTOKEN_CACHE_DIR`。
3. `graphrag init` 交互式问模型：`--model qwen-plus --embedding text-embedding-v3` + 喂空行 stdin。
4. settings 新 schema `completion_models:/embedding_models:`（非 2.x 的 `models:`），默认无 `api_base` → patch 注入百炼端点。
5. `graphrag query` 问题是**位置参数**（非 `--query`）；`--method local`；`--response-type` 设简洁。
6. `DG_CORE_DIR` 认准 `rag-L1/reproduce`（最终版 dg_core）；env.sh 自愈错值；驱动也自解析不依赖 env 传递。
7. 拉代码开学术加速、跑查询关加速（查询走百炼）。评测器用 `rag` 环境 python（依赖 lightrag）。
