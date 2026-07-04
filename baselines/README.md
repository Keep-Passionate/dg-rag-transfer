# DocBench 对比基线（IP&M 版新增） —— 规划与"各证明什么"

固定数据集 = DocBench，固定生成器 = qwen-plus（与主实验同一模型/端点/评测器，唯一变量是"怎么拿到答案"）。
三个基线各回答一个"审稿人必问"的反问，共同把 DG-RAG 的增益**从"可能只是检索没调好"里摘出来**。

## 三个基线，各证明什么
| 基线 | 做法 | 回答的反问 | 期望结果 / 论点 |
|---|---|---|---|
| **Long-context（无检索）** | 不检索，把整篇文档全文（截到上下文预算）直接喂给 LLM 问 | "把全文都给模型，不就行了？" | 全局属性题**照样答不好**——证明盲区**不是检索覆盖率**问题，长上下文也补不回索引丢掉的量 |
| **BM25 稀疏检索 RAG** | 切块→BM25 词法检索 top-k→生成（生成器同 vanilla） | "是不是稠密 embedding 的锅，换词法检索就好？" | 盲区**跨检索范式**依然在——证明不是某种 embedding 的偶然缺陷 |
| **PAL / Program-of-Thoughts** | LLM **每题现写 Python** 在文档文本上计算，再执行取值 | "让 LLM 自己写代码算，不比你固定算子强？" | PAL 是**最强概念对手**，但①每题一次 LLM 调用（贵、慢）②代码有非零出错率③不可复现；DG-RAG **零 LLM、确定性、可复现**，达到相当或更好 = 核心卖点 |

**一条递进故事**：`base(dense) ≈ BM25 ≤ long-context < PAL < DG-RAG`。
前三个说明"检索/长上下文都补不回全局属性"，PAL 说明"就算让 LLM 现写代码，固定确定性算子也不输、且零成本"。

## 为什么是这三个（检索/思考过的替代方案）
- **RAPTOR / 层级摘要**：概念上我们已论证"摘要有损、为检索而建"，且无现成易接入代码 → 留作 related work 讨论，不单独跑。
- **PDFTriage（Adobe）**：LLM triage 选 frame，是"LLM 路由"对手，但需适配其 pipeline + 非商用许可 → 可选、成本高，先不做（见交接 24）。
- **加大 top-k / 检索全部**：被 long-context 覆盖（更极端），可作 long-context 的一个点，不单列。
- **Reranker RAG**：主实验已有 `ENABLE_RERANK`，属"更好检索仍补不回"的又一证据，需要时零成本补一行，不单列。
- **Tool-use / function-calling agent**：本质=我们方法 + LLM 路由（=已有的 `ENABLE_NEURAL_ROUTING` 选项），与 PAL 论点重叠 → 用 PAL 代表"LLM-in-the-loop 计算"这一类。
- **结论**：这三个是覆盖"检索范式 / 长上下文 / LLM 现写代码"三条最常见质疑的最小充分集；PAL 是头条。**没有更好的必做项，先从 Long-context 开跑（最简单、最省）。**

## 文件
- `base_common.py` —— 共享：读 DocBench 文档全文（PyMuPDF）、LLM 客户端（与主实验同端点/模型）、写 `qa_results_<name>.json`。
- `run_longcontext.py` / `run_bm25.py` / `run_pal.py` —— 三基线驱动，逐题作答、写结果（可续跑）。
- `compare_baselines.py` —— 读评测 JSON，按 method 在 **meta-data 子集**报准确率（对齐主表）。
- `run_baselines.sh longcontext|bm25|pal|all [N]` —— 一条龙：跑基线 → 评测（复用主仓评测器）→ 对比。

## 运行（服务器）
```bash
cd /root/autodl-tmp/dg-rag-transfer && git pull
/root/miniconda3/envs/rag/bin/pip install -q rank_bm25         # 仅 BM25 需要
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
bash baselines/run_baselines.sh longcontext         # 先跑最简单的
bash baselines/run_baselines.sh all                 # 三个都跑 + 评测 + 对比
```
口径：结果写进各 doc 目录的 `qa_results_<name>.json`，评测器 glob 判分，`compare_baselines.py` 报 meta 准确率，填 IP&M 版 `tab:baselines`。
