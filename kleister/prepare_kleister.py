"""Kleister-NDA -> DocBench 式布局 + dg_core.parse 可作用子集统计（零 API 成本 = 探针）。

Kleister（ICDAR'21）每个 split 目录（train / dev-0 / test-A）是行对齐的两文件：
  in.tsv[.xz]   每行一篇文档：col0=filename，末列=OCR 全文（中间可能有 keys 列）。
  expected.tsv  每行一篇文档：空格分隔的 `key=value`（value 下划线连接），即金标要抽的字段。
NDA 实体键：effective_date / jurisdiction / party / term。

本脚本（探针，不花一分钱）：
  1) 对每篇文档、每个金标键，套成一句自然问题（忠于 Kleister 任务 = 抽该字段的值）。
  2) 跑 `dg_core.parse(q)` 统计命中——**并按键(effective_date/party/...)分桶**，看现有 4 算子代数覆盖到哪。
  3) 落地 DocBench 式目录 `<out>/<docid>/{<docid>.txt 或软链 PDF, <docid>_qa.jsonl}`，
     query.py / llm_answer_evaluator.py 零改动复用；产 kleister_manifest.json（按命中降序）。

诚实预期（已核对 reproduce/dg_core.py 文法）：现有算子里只有 `_RE_DATE_Q -> Lookup("date")`
会对 effective_date 触发；party / jurisdiction / term **不解析**。若探针证实如此：
  - 现有算子不改跑 effective_date 子集 = **纯泛化(generalization)**；
  - 为 party/jurisdiction/term 新增确定性 typed extractor = **可扩展性(extensibility)**，另列声明。

用法：python prepare_kleister.py --src /root/autodl-tmp/kleister-nda
"""
import argparse
import json
import lzma
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 忠于 Kleister 任务（抽字段值）的问题模板；只有能被 dg_core 现有文法解析的才会命中。
_Q_TEMPLATES = {
    "effective_date": "When did this agreement become effective? What is the effective date of this agreement?",
    "party": "Who are the parties to this agreement?",
    "jurisdiction": "What is the governing law or jurisdiction of this agreement?",
    "term": "What is the term or duration of this agreement?",
}


def _q_for(key: str) -> str:
    return _Q_TEMPLATES.get(key, f"What is the {key.replace('_', ' ')} of this document?")


def _read_lines(path: Path):
    if path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    return path.read_text(encoding="utf-8").splitlines()


def _find_in_tsv(split_dir: Path):
    for name in ("in.tsv", "in.tsv.xz"):
        p = split_dir / name
        if p.exists():
            return p
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="kleister-nda 克隆根目录（含 train/ dev-0/ test-A/）")
ap.add_argument("--out", default="/root/autodl-tmp/Kleister_subset")
ap.add_argument("--dg-core", default="/root/autodl-tmp/rag-L1/reproduce")
ap.add_argument("--splits", default="dev-0,test-A,train",
                help="要处理的 split（逗号分隔，只处理同时有 in.tsv+expected.tsv 的）")
a = ap.parse_args()

sys.path.insert(0, a.dg_core)
try:
    import dg_core
    parse_fn = dg_core.parse
except Exception as e:                      # parse 纯语法、不依赖文档
    print("!! dg_core.parse 不可用:", e)
    parse_fn = None

src = Path(a.src)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
allpdf = {p.stem: p for p in src.rglob("*.pdf")}
allpdf.update({p.name: p for p in src.rglob("*.pdf")})
print(f"仓内 PDF: {len(set(map(str, src.rglob('*.pdf'))))} 个"
      + ("（无 PDF -> 走 in.tsv 文本模式）" if not allpdf else ""))

man = []
by_key_total = Counter()
by_key_hit = Counter()
n_docs = n_q = n_hit = 0

for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
    sd = src / split
    in_tsv = _find_in_tsv(sd)
    exp = sd / "expected.tsv"
    if not (in_tsv and exp.exists()):
        print(f"跳过 split {split}（缺 in.tsv 或 expected.tsv）")
        continue
    ins = _read_lines(in_tsv)
    exps = _read_lines(exp)
    # 去掉可能的表头（首行不含制表符或明显是列名）
    if ins and "\t" not in ins[0] and len(ins) == len(exps) + 1:
        ins = ins[1:]
    n = min(len(ins), len(exps))
    print(f"[{split}] {n} 篇（in.tsv {len(ins)} / expected.tsv {len(exps)}）")
    for i in range(n):
        cols = ins[i].split("\t")
        fname = cols[0].strip() if cols else f"{split}_{i}"
        text = cols[-1] if len(cols) >= 2 else ""
        # 金标：`key=value` 空格分隔；同键可多值（如多个 party=）；value 用下划线连接。
        gold = {}                              # key -> [values]（保序）
        for tok in exps[i].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                gold.setdefault(k, []).append(v.replace("_", " ").strip())
        keys = list(gold.keys())               # 保序去重
        if not keys:
            continue
        did = f"k_{split.replace('-', '')}_{i:04d}"
        d = out / did
        d.mkdir(exist_ok=True)
        pdf = allpdf.get(Path(fname).stem) or allpdf.get(fname)
        if pdf:                                # 有 PDF：软链进来，index.py 可解析
            dst = d / pdf.name
            if not dst.exists():
                try:
                    dst.symlink_to(pdf.resolve())
                except Exception:
                    shutil.copy(pdf, dst)
        elif text:                             # 无 PDF：落地 OCR 文本（文本模式回退）
            (d / f"{did}.txt").write_text(text.replace("\\n", "\n"), encoding="utf-8")
        hits = 0
        with open(d / f"{did}_qa.jsonl", "w", encoding="utf-8") as fo:
            for key in keys:
                q = _q_for(key)
                ans = ", ".join(dict.fromkeys(gold[key]))   # 多值(如多个 party)去重合并 = 金标
                fired = bool(parse_fn(q)) if parse_fn else False
                by_key_total[key] += 1
                by_key_hit[key] += int(fired)
                hits += fired
                n_q += 1
                n_hit += fired
                fo.write(json.dumps({
                    "question": q,
                    "answer": ans,                       # query.py 会把它写进 correct_answer 供评测判分
                    "type": f"kleister:{key}",
                    "dg_parse_hit": fired,
                    "kleister_key": key,
                    "source_file": fname,
                }, ensure_ascii=False) + "\n")
        n_docs += 1
        man.append({"doc_id": did, "split": split, "source_file": fname,
                    "questions": len(keys), "parse_hits": hits, "has_pdf": bool(pdf)})

man.sort(key=lambda r: -r["parse_hits"])
json.dump(man, open(out / "kleister_manifest.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n================ 探针结果（零 API） ================")
print(f"落地 {n_docs} 篇 / {n_q} 题；dg_core.parse 命中 {n_hit} 题；"
      f"命中≥1 的文档 {sum(1 for r in man if r['parse_hits'])} 篇；"
      f"带 PDF 的文档 {sum(1 for r in man if r['has_pdf'])} 篇")
print("---- 按实体键分桶（命中/总数）----")
for key in sorted(by_key_total, key=lambda k: -by_key_hit[k]):
    print(f"  {key:16s}: {by_key_hit[key]:4d} / {by_key_total[key]:4d} 命中"
          + ("   <- 现有算子覆盖" if by_key_hit[key] else "   <- 现有算子不覆盖(需 typed extractor)"))
print("manifest（按命中降序）:", out / "kleister_manifest.json")
print("提示：命中主要集中在 effective_date 即符合代码预期；命中子集够大再跑 run_kleister.sh。")
