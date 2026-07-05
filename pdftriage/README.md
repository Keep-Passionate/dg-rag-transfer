# PDFTriage 探针（第二数据集候选 · 最可信）

PDFTriage（Adobe，EMNLP'24 Industry，`github.com/adobe-research/pdftriage`，908 题/82 篇）
含 **Structure / Extraction** 题类——现有基准里离 DocBench-meta 最近的"结构/全局属性"题，
正是 DG 的主场（检索够不到、程序能算），与 Kleister 的可检索跨度本质不同。已发表基准=比自建可信。

## 探针（零 API）
```bash
source /etc/network_turbo
cd /root/autodl-tmp && git clone https://github.com/adobe-research/pdftriage.git \
  || (cd pdftriage && git pull)
cd /root/autodl-tmp/dg-rag-transfer && git pull
/root/miniconda3/envs/rag/bin/python pdftriage/probe_pdftriage.py --src /root/autodl-tmp/pdftriage
```
输出按 category 分桶报 `dg_core.parse` 命中率，重点看 **Structure/Extraction 子集大小 + 命中**。
- 够大且命中集中在结构/抽取 → 建索引跑 base vs +DG（预期像 DocBench 大增益）。
- 若 Structure 太小（PDFTriage Structure 仅 ~3.7%≈34 题）→ 用自建"全局属性 QA"集（独立客观金标）补 n。

若脚本找不到题目文件，把 `ls -R /root/autodl-tmp/pdftriage | head -50` 贴回，我按真实格式调。
