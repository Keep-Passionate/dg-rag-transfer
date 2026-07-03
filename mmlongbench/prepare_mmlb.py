"""MMLongBench-Doc -> DocBench 式布局 + dg_core.parse 可作用子集统计（零 API 成本）。

输入：GitHub 克隆的 MMLongBench-Doc 仓库（data/samples.json + data/documents/*.pdf）。
输出：/root/autodl-tmp/MMLB_subset/<mNNN>/{<pdf>, <mNNN>_qa.jsonl} —— 与 DocBench 同布局，
主仓 reproduce/query.py（读 <目录名>_qa.jsonl）与 llm_answer_evaluator.py 零改动复用。
另产 mmlb_manifest.json（按 dg_core.parse 命中数降序，跑 25 篇时优先命中多的）。

用法：python prepare_mmlb.py --src /root/autodl-tmp/MMLB_raw
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="MMLB_raw 目录（含 data/samples.json）")
ap.add_argument("--out", default="/root/autodl-tmp/MMLB_subset")
ap.add_argument("--dg-core", default="/root/autodl-tmp/rag-L1/reproduce")
a = ap.parse_args()

sys.path.insert(0, a.dg_core)
try:
    import dg_core
    parse_fn = dg_core.parse
except Exception as e:                    # parse 纯语法、不依赖文档，但保险起见容错
    print("!! dg_core.parse 不可用:", e)
    parse_fn = None

src = Path(a.src)
samples, dockey = None, None
for f in list(src.rglob("*.jsonl")) + list(src.rglob("*.json")):
    try:
        objs = ([json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if f.suffix == ".jsonl" else json.loads(f.read_text(encoding="utf-8")))
        if isinstance(objs, dict):
            objs = objs.get("data")
    except Exception:
        continue
    if isinstance(objs, list) and objs and isinstance(objs[0], dict) and "question" in objs[0]:
        dockey = next((k for k in ("doc_id", "pdf", "document", "doc", "file", "doc_name")
                       if k in objs[0]), None)
        if dockey:
            print(f"标注: {f} ({len(objs)} 条, doc字段={dockey})")
            samples = objs
            break
if not samples:
    print("!! 没找到含 question+doc 字段的标注文件——把 --src 下 ls -R 贴回")
    sys.exit(2)

allpdf = list(src.rglob("*.pdf"))
pdfs = {}
for p in allpdf:
    pdfs[p.name] = p
    pdfs[p.stem] = p
print(f"PDF {len(allpdf)} 个")

bydoc = defaultdict(list)
for o in samples:
    bydoc[str(o[dockey])].append(o)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
man, miss = [], 0
for i, (doc, qs) in enumerate(sorted(bydoc.items())):
    p = pdfs.get(doc) or pdfs.get(Path(doc).name) or pdfs.get(Path(doc).stem)
    if not p:
        miss += 1
        continue
    did = f"m{i:03d}"
    d = out / did
    d.mkdir(exist_ok=True)
    dst = d / p.name
    if not dst.exists():
        try:
            dst.symlink_to(p.resolve())
        except Exception:
            shutil.copy(p, dst)
    hits = 0
    with open(d / f"{did}_qa.jsonl", "w", encoding="utf-8") as fo:
        for o in qs:
            q = str(o.get("question", ""))
            fired = bool(parse_fn(q)) if parse_fn else False
            hits += fired
            fo.write(json.dumps({
                "question": q,
                "answer": str(o.get("answer", "")),
                "type": str(o.get("evidence_sources") or o.get("answer_format") or ""),
                "dg_parse_hit": fired,
                "evidence_pages": str(o.get("evidence_pages", "")),
            }, ensure_ascii=False) + "\n")
    man.append({"doc_id": did, "pdf": p.name, "questions": len(qs), "parse_hits": hits})

man.sort(key=lambda r: -r["parse_hits"])
json.dump(man, open(out / "mmlb_manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tq = sum(r["questions"] for r in man)
th = sum(r["parse_hits"] for r in man)
print(f"落地 {len(man)} 篇(缺PDF {miss}) / {tq} 题; dg_core.parse 命中 {th} 题; "
      f"命中≥1 的文档 {sum(1 for r in man if r['parse_hits'])} 篇")
print("manifest(按 parse 命中降序):", out / "mmlb_manifest.json")
