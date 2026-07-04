# Kleister-NDA 跨数据集迁移（第二数据集主攻）

固定骨干 RAG-Anything，换到 Kleister-NDA（ICDAR'21，法律长合同 key-field 抽取）上跑 base vs +DG，
证 DG 增益**跨数据集 + 跨领域**（非过拟合 DocBench）。见交接 24。

## ⚠️ 先探针再花钱（纪律）
现有 `dg_core` 4 算子代数（Count/Locate/Extract/Lookup）里，**只有 `Lookup("date")` 会对
`effective_date` 触发**（已核对 `reproduce/dg_core.py` 的 `_RE_DATE_Q`）；NDA 的 `party /
jurisdiction / term` 现有文法**不解析**。所以：
1. 先跑 `prepare_kleister.py`（**零 API**）看命中子集有多大、集中在哪个键。
2. 命中够大 → `run_kleister.sh` 建索引跑 ±DG。
3. 两种声明分开写：现有算子不改跑 effective_date = **纯泛化**；给 party/jurisdiction/term
   新增确定性 typed extractor = **可扩展性**（次级贡献），别混。

## 文件
- `prepare_kleister.py` — Kleister split(train/dev-0/test-A) → DocBench 式布局
  （`Kleister_subset/<docid>/{pdf 或 txt, <docid>_qa.jsonl}`）+ `dg_core.parse` 按键分桶统计（零 API）；
  产 `kleister_manifest.json`（按命中降序）。**这一步就是探针。**
- `run_kleister.sh smoke|25|all` — 建图→查询 base/dg→评测→对比，可续跑。
- `compare_kleister.py` — DG触发子集/未触发/overall + 按实体键分桶 + McNemar（零 LLM）。

## 口径
base = paperbase（`ENABLE_DG_CORE=false`，其余增强全关）；dg = 仅 `ENABLE_DG_CORE=true`
（+`PARSE_OUTPUT_DIR`）。唯一变量是 DG，与主实验/GraphRAG/vanilla 同一套模型与评测器。

## 运行（服务器）
```bash
source /etc/network_turbo                       # 仅 git/clone 用，跑实验前会关
cd /root/autodl-tmp && git clone https://github.com/applicaai/kleister-nda.git \
  || (cd kleister-nda && git pull)
cd /root/autodl-tmp/dg-rag-transfer && git pull
# ① 探针（零成本）
/root/miniconda3/envs/rag/bin/python kleister/prepare_kleister.py --src /root/autodl-tmp/kleister-nda
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # 关学术加速（查询走百炼）
# ② 命中够大再跑（先 smoke 3 篇打通，再 all）
bash kleister/run_kleister.sh smoke
bash kleister/run_kleister.sh all
```
另开终端看弹幕：`tail -f /root/autodl-tmp/kleister_all.log`
