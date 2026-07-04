# DUDE 小样本探针（次选第二数据集，先探针）

DUDE（ICCV'23，5000 篇多页 PDF / ~18.7K 题）结构最像 DocBench（答案类型含
extractive / dates / lists / **not-answerable**，甚至呼应我们的弃权叙事），但**视觉丰富、
无独立 metadata 类**，可作用子集不确定（=MMLB 同款风险）。交接 24 §4：**必须先探针**，
命中子集够大才建索引。

## 小样本测试（零 API，只判覆盖）
```bash
cd /root/autodl-tmp/dg-rag-transfer && git pull
source /etc/network_turbo                        # HF 下载用
/root/miniconda3/envs/rag/bin/python dude/probe_dude.py --limit 300 --split val
```
- 直接对 DUDE 问题跑 `dg_core.parse`，按 `answer_type` 分桶报命中率，不建索引、不花钱。
- 需 `pip install datasets`；`--limit` 控小样本题数（默认流式取前 300，省下载）。
- 判读：命中子集占比够大且集中在 extractive/counting/date → 再照 kleister 那套建索引跑 ±DG；
  若近零命中（像 MMLB）→ 记 Limitation，不投入。

eval 仓：`github.com/Jordy-VL/DUDEeval`（真要跑 ±DG 时再接）。
