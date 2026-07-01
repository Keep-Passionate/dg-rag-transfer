#!/usr/bin/env python
"""GraphRAG 迁移驱动：per-document 索引 + 逐题查询 ±augment，产出可被主仓评测器判分的结果。

对每篇（子集 doc_id）：
  1) content_list → GraphRAG 输入 txt（两骨干吃同一份 MinerU 解析，公平+省钱）
  2) graphrag init（生成版本匹配的默认配置+prompts）→ patch_settings 改成 qwen/dashscope → graphrag index
  3) 逐道 meta 题查两遍：base=原题、dg=augment(原题)；门控弃权(aug_q==question)时 dg 复用 base（省一半查询）
  4) 结果写到 <eval_root>/<doc_id>/qa_results_graphrag_base.json 与 ..._graphrag_dg.json
     （记录里 question=原题，以对金标；同目录放 PDF 同名占位符，让评测器正确推断 doc_id）

可续跑：索引产物已在则跳过建图；两个结果文件都在则跳过该篇查询（--reindex/--requery 强制）。
内容审查(dashscope data_inspection_failed)或建图失败：记录并跳过该篇，不让一篇拖垮整批。

环境变量（见 env.sh）：DG_CORE_DIR PARSE_OUTPUT_DIR GRAPHRAG_API_KEY DASHSCOPE_API_KEY
用法示例：
  python run_graphrag_docbench.py --limit 2                 # smoke：top25 前 2 篇
  python run_graphrag_docbench.py --docs 0,102,119          # 指定篇
  python run_graphrag_docbench.py --manifest top25.json     # 全 25 篇
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# GraphRAG 3.x 经 litellm 调模型；litellm 启动默认联网拉 GitHub 价目表，国内会卡死。
# 子进程继承本进程环境，这里兜底设一遍（env.sh 里也有，双保险）。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent

# 接地层(dg_core)目录：不依赖 env.sh 的导出/子进程继承（那条链太脆，断了就回退 Windows 默认路径必崩）。
# 这里自己解析：DG_CORE_DIR 环境变量优先，否则用服务器已知最终版路径；找不到就明确报错。
_DG_CANDIDATES = [os.getenv("DG_CORE_DIR", ""), "/root/autodl-tmp/rag-L1/reproduce"]
DG_CORE_DIR = next((p for p in _DG_CANDIDATES if p and (Path(p) / "dg_core.py").exists()), "")
if not DG_CORE_DIR:
    sys.exit("!! 找不到 dg_core.py。请 export DG_CORE_DIR=<含 dg_core.py 的 reproduce 目录>。"
             f"已尝试: {[p for p in _DG_CANDIDATES if p]}")
os.environ["DG_CORE_DIR"] = DG_CORE_DIR            # 供 dg_augmenter/dg_core 内部读取，保持一致
sys.path.insert(0, DG_CORE_DIR)
print(f"[dg_core] 使用接地层目录: {DG_CORE_DIR}", flush=True)

sys.path.insert(0, str(HERE))            # build_input
sys.path.insert(0, str(HERE.parent))     # dg_augmenter
import build_input  # noqa: E402
import dg_augmenter  # noqa: E402

RESP_RE = re.compile(r"(?:Local|Global|Basic|DRIFT)\s+Search\s+Response:\s*", re.I)
# 滤掉日志噪声：级别前缀行、litellm 那两行 warning、以及形如 "17:23:35 - ..." 的时间戳日志行
LOG_RE = re.compile(r"^\s*(INFO|SUCCESS|WARNING|WARN|ERROR|DEBUG)\b|LiteLLM|^\d{1,2}:\d{2}:\d{2}\b")

# meta 题金标多为短答案（人名/页码/计数）。默认让 GraphRAG 简洁作答（base/dg 同等施加，公平）。
RESPONSE_TYPE = ("Single short sentence; if the answer is a name, number, date, or page, "
                 "give only that value.")


def log(msg):
    print(msg, flush=True)


def load_questions(docbench, doc_id, only_meta=True):
    """only_meta=True 只取 meta-data 题；False 取全部题型（用于 overall/非回归）。"""
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


def parse_answer(stdout):
    text = stdout or ""
    m = RESP_RE.search(text)
    if m:
        return text[m.end():].strip()
    keep = [ln for ln in text.splitlines() if ln.strip() and not LOG_RE.match(ln)]
    return "\n".join(keep).strip()


def index_ready(root):
    # 完整索引才算就绪：不能只看有没有 parquet（嵌入步失败会留下"实体在、lancedb 缺"的半成品，
    # 让 local search 查询时 read_parquet 崩、记空答）。要求晚阶段的 community_reports + lancedb 都在。
    out = Path(root) / "output"
    if not (out / "entities.parquet").exists() or not (out / "community_reports.parquet").exists():
        return False
    lancedb = out / "lancedb"
    return lancedb.exists() and any(lancedb.iterdir())


def ensure_index(root, pdf, doc_id, content_lists, reindex):
    root = Path(root)
    if index_ready(root) and not reindex:
        log(f"[{doc_id}] 索引已存在，跳过建图")
        return True
    root.mkdir(parents=True, exist_ok=True)
    # 1) graphrag init（生成默认 settings.yaml + prompts + .env）
    # GraphRAG 3.x 的 init 交互式追问 chat/embedding 模型名——用 --model/--embedding 直接给定，
    # 并喂空行到 stdin，确保非交互、不卡在提示符（patch_settings 随后还会强制覆盖）。
    subprocess.run(["graphrag", "init", "--root", str(root), "--force",
                    "--model", "qwen-plus", "--embedding", "text-embedding-v3"],
                   check=False, capture_output=True, text=True, input="\n" * 10)
    # 2) patch 成 qwen/dashscope
    r = subprocess.run([sys.executable, str(HERE / "patch_settings.py"), str(root / "settings.yaml")],
                       capture_output=True, text=True)
    log(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        log(f"[{doc_id}] !! patch_settings 失败：{r.stderr.strip()}")
        return False
    # 3) 输入 txt（content_list 优先）
    inp = root / "input"
    inp.mkdir(parents=True, exist_ok=True)
    cl = Path(content_lists) / (Path(pdf).stem + "_content_list.json")
    info = build_input.build_input(pdf, str(inp / f"{doc_id}.txt"),
                                   str(cl) if cl.exists() else None)
    log(f"[{doc_id}] 输入 {info['source']} chars={info['chars']}")
    # 4) 建图
    log(f"[{doc_id}] graphrag index 开始 …")
    t0 = time.time()
    r = subprocess.run(["graphrag", "index", "--root", str(root)], text=True)
    ok = r.returncode == 0 and index_ready(root)
    log(f"[{doc_id}] graphrag index {'完成' if ok else '失败'}（{time.time() - t0:.0f}s）")
    return ok


def graphrag_query(root, method, q, response_type=RESPONSE_TYPE):
    # GraphRAG 3.x：问题是**位置参数**，不是 --query。用法 `graphrag query --root X --method local "问题"`。
    r = subprocess.run(["graphrag", "query", "--root", str(root), "--method", method,
                        "--response-type", response_type, q],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # 不再静默返回空：把失败原因打出来（内容审查/认证/参数等），便于定位。
        err = (r.stderr or r.stdout or "").strip().splitlines()
        log(f"    !! graphrag query 失败(rc={r.returncode}): {' '.join(err[-3:])[:200]}")
        return ""
    return parse_answer(r.stdout)


def write_results(eval_root, doc_id, pdf, base_recs, dg_recs):
    edir = Path(eval_root) / doc_id
    edir.mkdir(parents=True, exist_ok=True)
    placeholder = edir / Path(pdf).name          # 评测器只用文件名推断 doc_id，不读内容
    if not placeholder.exists():
        try:
            placeholder.symlink_to(pdf)
        except Exception:
            placeholder.write_text("", encoding="utf-8")
    json.dump(base_recs, open(edir / "qa_results_graphrag_base.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(dg_recs, open(edir / "qa_results_graphrag_dg.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def already_queried(eval_root, doc_id):
    edir = Path(eval_root) / doc_id
    return (edir / "qa_results_graphrag_base.json").exists() and \
           (edir / "qa_results_graphrag_dg.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "top25.json"))
    ap.add_argument("--docs", default="", help="逗号分隔 doc_id，覆盖 manifest")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 篇（smoke 用）")
    ap.add_argument("--method", default="local", choices=["local", "global", "basic", "drift"])
    ap.add_argument("--response-type", default=RESPONSE_TYPE, help="GraphRAG --response-type（简洁作答）")
    ap.add_argument("--docbench", default=os.getenv("DOCBENCH_ROOT", "/root/autodl-tmp/DocBench_subset"))
    ap.add_argument("--content-lists", default=os.getenv("PARSE_OUTPUT_DIR", "/root/autodl-tmp/content_lists"))
    ap.add_argument("--runs", default=os.getenv("GRAPHRAG_RUNS", "/root/autodl-tmp/graphrag_runs"))
    ap.add_argument("--eval-root", default=os.getenv("GRAPHRAG_EVAL", "/root/autodl-tmp/graphrag_eval"))
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--requery", action="store_true")
    ap.add_argument("--all-types", action="store_true",
                    help="跑全部题型(不只meta)，用于 overall + 非meta非回归；非meta题DG弃权→dg复用base不额外查询")
    ap.add_argument("--exclude", default=os.getenv("EXCLUDE_IDS", ""),
                    help="逗号分隔 doc_id，跳过（如内容审查必失败的敏感文档，省得全量跑每次重试白耗时）")
    ap.add_argument("--all-docs", action="store_true",
                    help="扫描 DocBench_subset 下全部文档(全量)，忽略 manifest/limit")
    a = ap.parse_args()

    if a.all_docs:
        base = Path(a.docbench)
        doc_ids = sorted((p.name for p in base.iterdir()
                          if p.is_dir() and (p / f"{p.name}_qa.jsonl").exists()),
                         key=lambda s: int(s) if s.isdigit() else 1 << 30)
    elif a.docs.strip():
        doc_ids = [d.strip() for d in a.docs.split(",") if d.strip()]
    else:
        man = json.load(open(a.manifest, encoding="utf-8"))
        doc_ids = [d["doc_id"] for d in man["docs"]]
    if a.limit:
        doc_ids = doc_ids[:a.limit]
    excl = {s.strip() for s in a.exclude.split(",") if s.strip()}
    if excl:
        doc_ids = [d for d in doc_ids if d not in excl]
        log(f"[exclude] 跳过 {sorted(excl)}")

    if not os.getenv("GRAPHRAG_API_KEY"):
        log("!! GRAPHRAG_API_KEY 未设置——请 source env.sh 或 export（=你的百炼 key）")
        sys.exit(1)

    log(f"==== GraphRAG 迁移：{len(doc_ids)} 篇（method={a.method}）====")
    stats = {"indexed": 0, "index_fail": 0, "queried": 0, "skipped": 0, "meta_q": 0, "dg_fire": 0}
    for i, doc_id in enumerate(doc_ids, 1):
        pdf = find_pdf(a.docbench, doc_id)
        if not pdf:
            log(f"[{i}/{len(doc_ids)}] doc {doc_id}: 找不到 PDF，跳过")
            stats["index_fail"] += 1
            continue
        log(f"\n[{i}/{len(doc_ids)}] doc {doc_id}  {Path(pdf).name}")
        root = Path(a.runs) / doc_id
        # 先看索引是否已完整：若否，ensure_index 会（借 LLM 缓存）重建——这种情况旧的空答案已过期，必须重查。
        was_ready = index_ready(root)
        if not ensure_index(root, pdf, doc_id, a.content_lists, a.reindex):
            log(f"[{doc_id}] !! 建图失败，跳过本篇")
            stats["index_fail"] += 1
            continue
        stats["indexed"] += 1
        rebuilt = not was_ready          # 刚重建的篇：强制重查，覆盖旧空答
        # all-types 会把非 meta 题也补进结果，故此模式下不跳过（需重写全题型结果）
        if already_queried(a.eval_root, doc_id) and not rebuilt and not a.requery \
                and not a.reindex and not a.all_types:
            log(f"[{doc_id}] 结果已存在且索引完整，跳过查询")
            stats["skipped"] += 1
            continue

        qs = load_questions(a.docbench, doc_id, only_meta=not a.all_types)
        base_recs, dg_recs = [], []
        for j, item in enumerate(qs, 1):
            q, gold, qtype = item["question"], item["answer"], item.get("type", "")
            stats["meta_q"] += 1
            base = graphrag_query(root, a.method, q, a.response_type)
            aug_q, meta = dg_augmenter.augment(q, pdf)
            if aug_q == q:
                dg = base                      # 弃权(含所有非meta题)→恒等回退，dg 复用 base，不额外查询
            else:
                stats["dg_fire"] += 1
                dg = graphrag_query(root, a.method, aug_q, a.response_type)
            base_recs.append({"question": q, "answer": base, "correct_answer": gold, "type": qtype})
            dg_recs.append({"question": q, "answer": dg, "correct_answer": gold, "type": qtype,
                            "dg_used": meta.get("dg_used"), "dg_kind": meta.get("dg_kind")})
            log(f"    Q{j}/{len(qs)} [{qtype}] dg_used={meta.get('dg_used')} kind={meta.get('dg_kind')}")
        write_results(a.eval_root, doc_id, pdf, base_recs, dg_recs)
        stats["queried"] += 1
        log(f"[{doc_id}] 已写结果（meta {len(qs)} 题）")

    log(f"\n==== 完成 {stats} ====")
    log(f"下一步评测：python <主仓>/reproduce/llm_answer_evaluator.py --qa-data-dir {a.eval_root} "
        f"-o {a.eval_root}/_eval --api-key $DASHSCOPE_API_KEY "
        f"--base-url https://dashscope.aliyuncs.com/compatible-mode/v1")


if __name__ == "__main__":
    main()
