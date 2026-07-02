#!/usr/bin/env python
"""Vanilla/native RAG 驱动：每篇建向量索引（嵌入切块，无 LLM）+ 逐题 base/dg 查询，
产出与 GraphRAG 版同款 qa_results_vanilla_{base,dg}.json（复用主仓评测器 + compare_graphrag）。

非 meta 题 DG 弃权 → aug_q==question → dg 复用 base，不额外查询。可续跑、失败跳过、进度弹幕。
用法示例：
  python run_vanilla_docbench.py --limit 2                      # smoke
  python run_vanilla_docbench.py --manifest ../graphrag/top25.json --all-types
  python run_vanilla_docbench.py --all-docs --all-types         # 全量全题型
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DASHSCOPE_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
HERE = Path(__file__).resolve().parent

# dg_core 目录自解析（同 graphrag 驱动，不依赖 env 传递）
_DG_CANDIDATES = [os.getenv("DG_CORE_DIR", ""), "/root/autodl-tmp/rag-L1/reproduce"]
DG_CORE_DIR = next((p for p in _DG_CANDIDATES if p and (Path(p) / "dg_core.py").exists()), "")
if not DG_CORE_DIR:
    sys.exit("!! 找不到 dg_core.py，请 export DG_CORE_DIR=<含 dg_core.py 的 reproduce 目录>")
os.environ["DG_CORE_DIR"] = DG_CORE_DIR
sys.path.insert(0, DG_CORE_DIR)
print(f"[dg_core] {DG_CORE_DIR}", flush=True)

sys.path.insert(0, str(HERE))                       # vanilla_rag
sys.path.insert(0, str(HERE.parent))                # dg_augmenter
sys.path.insert(0, str(HERE.parent / "graphrag"))   # build_input
import vanilla_rag as vr  # noqa: E402
import dg_augmenter  # noqa: E402
import build_input  # noqa: E402


def log(m):
    print(m, flush=True)


def load_questions(docbench, doc_id, only_meta=True):
    qa = Path(docbench) / doc_id / f"{doc_id}_qa.jsonl"
    out = []
    if not qa.exists():
        return out
    for line in qa.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type", "")
        if only_meta and t != "meta-data":
            continue
        out.append({"question": o.get("question", ""), "answer": o.get("answer", ""), "type": t})
    return out


def find_pdf(docbench, doc_id):
    hits = glob.glob(str(Path(docbench) / doc_id / "*.pdf"))
    return hits[0] if hits else None


def content_text(pdf, content_lists):
    cl = Path(content_lists) / (Path(pdf).stem + "_content_list.json")
    if cl.exists():
        items = [x for x in json.loads(cl.read_text(encoding="utf-8")) if isinstance(x, dict)]
        return build_input.serialize_content_list(items)
    try:
        return build_input.pdf_fulltext(pdf)          # 回退 PyMuPDF
    except Exception:
        return ""


def write_results(eval_root, doc_id, pdf, base_recs, dg_recs):
    edir = Path(eval_root) / doc_id
    edir.mkdir(parents=True, exist_ok=True)
    ph = edir / Path(pdf).name                        # 评测器只用文件名推 doc_id
    if not ph.exists():
        try:
            ph.symlink_to(pdf)
        except Exception:
            ph.write_text("", encoding="utf-8")
    json.dump(base_recs, open(edir / "qa_results_vanilla_base.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(dg_recs, open(edir / "qa_results_vanilla_dg.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def already_queried(eval_root, doc_id, expect_n):
    """结果文件存在且题数不少于本次应查题数才算查过——使 --all-types 断点续跑也能正确跳过。"""
    e = Path(eval_root) / doc_id
    fb = e / "qa_results_vanilla_base.json"
    fd = e / "qa_results_vanilla_dg.json"
    if not (fb.exists() and fd.exists()):
        return False
    try:
        return len(json.loads(fb.read_text(encoding="utf-8"))) >= expect_n
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "graphrag" / "top25.json"))
    ap.add_argument("--docs", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-docs", action="store_true")
    ap.add_argument("--all-types", action="store_true")
    ap.add_argument("--exclude", default=os.getenv("EXCLUDE_IDS", ""))
    ap.add_argument("--docbench", default=os.getenv("DOCBENCH_ROOT", "/root/autodl-tmp/DocBench_subset"))
    ap.add_argument("--content-lists", default=os.getenv("PARSE_OUTPUT_DIR", "/root/autodl-tmp/content_lists"))
    ap.add_argument("--runs", default=os.getenv("VANILLA_RUNS", "/root/autodl-tmp/vanilla_runs"))
    ap.add_argument("--eval-root", default=os.getenv("VANILLA_EVAL", "/root/autodl-tmp/vanilla_eval"))
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--requery", action="store_true")
    ap.add_argument("--top-k", type=int, default=vr.TOP_K)
    a = ap.parse_args()

    if a.all_docs:
        base = Path(a.docbench)
        doc_ids = sorted((p.name for p in base.iterdir()
                          if p.is_dir() and (p / f"{p.name}_qa.jsonl").exists()),
                         key=lambda s: int(s) if s.isdigit() else 1 << 30)
    elif a.docs.strip():
        doc_ids = [d.strip() for d in a.docs.split(",") if d.strip()]
    else:
        doc_ids = [d["doc_id"] for d in json.load(open(a.manifest, encoding="utf-8"))["docs"]]
    if a.limit:
        doc_ids = doc_ids[:a.limit]
    excl = {s.strip() for s in a.exclude.split(",") if s.strip()}
    if excl:
        doc_ids = [d for d in doc_ids if d not in excl]
        log(f"[exclude] 跳过 {sorted(excl)}")

    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("GRAPHRAG_API_KEY")):
        sys.exit("!! DASHSCOPE_API_KEY 未设置——请 source ../graphrag/env.sh")

    log(f"==== Vanilla RAG：{len(doc_ids)} 篇（top_k={a.top_k}）====")
    st = {"indexed": 0, "index_fail": 0, "queried": 0, "skipped": 0, "q": 0, "dg_fire": 0}
    for i, doc_id in enumerate(doc_ids, 1):
        pdf = find_pdf(a.docbench, doc_id)
        if not pdf:
            log(f"[{i}/{len(doc_ids)}] doc {doc_id}: 无 PDF，跳过")
            st["index_fail"] += 1
            continue
        log(f"\n[{i}/{len(doc_ids)}] doc {doc_id}  {Path(pdf).name}")
        run_dir = Path(a.runs) / doc_id
        was_ready = vr.index_ready(run_dir)
        if not was_ready or a.reindex:
            try:
                t0 = time.time()
                n = vr.build_index(content_text(pdf, a.content_lists), run_dir)
                if n == 0:
                    raise RuntimeError("空文本，无 chunk")
                log(f"[{doc_id}] 索引 {n} chunks（{time.time() - t0:.0f}s）")
                st["indexed"] += 1
            except Exception as e:
                log(f"[{doc_id}] !! 建索引失败：{str(e)[:150]}")
                st["index_fail"] += 1
                continue
        else:
            log(f"[{doc_id}] 索引已存在，跳过")
        rebuilt = not was_ready
        qs = load_questions(a.docbench, doc_id, only_meta=not a.all_types)
        if already_queried(a.eval_root, doc_id, len(qs)) and not rebuilt and not a.requery:
            log(f"[{doc_id}] 结果已存在（题数齐），跳过查询")
            st["skipped"] += 1
            continue
        try:
            emb, chunks = vr.load_index(run_dir)
        except Exception as e:
            log(f"[{doc_id}] !! 读索引失败：{str(e)[:120]}")
            st["index_fail"] += 1
            continue

        base_recs, dg_recs = [], []
        for j, item in enumerate(qs, 1):
            q, gold, qt = item["question"], item["answer"], item.get("type", "")
            st["q"] += 1
            try:
                base = vr.answer(q, emb, chunks, a.top_k)
            except Exception as e:
                base = ""
                log(f"    Q{j} base 失败：{str(e)[:100]}")
            aug_q, meta = dg_augmenter.augment(q, pdf)
            if aug_q == q:
                dg = base
            else:
                st["dg_fire"] += 1
                try:
                    dg = vr.answer(aug_q, emb, chunks, a.top_k)
                except Exception as e:
                    dg = base
                    log(f"    Q{j} dg 失败：{str(e)[:100]}")
            base_recs.append({"question": q, "answer": base, "correct_answer": gold, "type": qt})
            dg_recs.append({"question": q, "answer": dg, "correct_answer": gold, "type": qt,
                            "dg_used": meta.get("dg_used"), "dg_kind": meta.get("dg_kind")})
            log(f"    Q{j}/{len(qs)} [{qt}] dg_used={meta.get('dg_used')} {meta.get('dg_kind')}")
        write_results(a.eval_root, doc_id, pdf, base_recs, dg_recs)
        st["queried"] += 1
        log(f"[{doc_id}] 已写结果（{len(qs)} 题）")

    log(f"\n==== 完成 {st} ====")


if __name__ == "__main__":
    main()
