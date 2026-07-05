"""Compare MMLongBench-Doc RAG-Anything variants after LLM judging.

Expected result files inside each MMLB document directory:
  qa_results_mmlb_base.json
  qa_results_mmlb_dg.json
  qa_results_mmlb_mm.json
  qa_results_mmlb_dg_mm.json

The evaluator names methods by stripping ``qa_results_`` and ``.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import comb
from typing import Callable, Dict, Iterable, List, Tuple

Key = Tuple[str, str]


def norm(text: str) -> str:
    return " ".join((text or "").split())


def mcnemar_p(base_lost: int, method_won: int) -> float:
    n = base_lost + method_won
    if n == 0:
        return 1.0
    k = min(base_lost, method_won)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def load_eval(path: str) -> Dict[str, Dict[Key, int]]:
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw
    out: Dict[str, Dict[Key, int]] = {}
    for row in rows:
        method = row.get("method")
        if not method:
            continue
        try:
            acc = int(row.get("accuracy"))
        except Exception:
            continue
        key = (row.get("doc_id", ""), norm(row.get("question", "")))
        out.setdefault(method, {})[key] = acc
    return out


def doc_pdf_name(folder: str) -> str:
    return next((f for f in os.listdir(folder) if f.lower().endswith(".pdf")), os.path.basename(folder))


def load_trigger_flags(mmlb: str, filename: str, fields: Iterable[str]) -> Dict[Key, bool]:
    flags: Dict[Key, bool] = {}
    for path in glob.glob(os.path.join(mmlb, "m*", filename)):
        folder = os.path.dirname(path)
        doc_id = doc_pdf_name(folder)
        try:
            rows = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for row in rows:
            flags[(doc_id, norm(row.get("question", "")))] = any(bool(row.get(field)) for field in fields)
    return flags


def summarize_pair(
    acc: Dict[str, Dict[Key, int]],
    base_method: str,
    method: str,
    label: str,
    predicate: Callable[[Key], bool],
) -> None:
    keys = [k for k in acc.get(method, {}) if k in acc.get(base_method, {}) and predicate(k)]
    if not keys:
        print(f"  {label:<18} n=0")
        return
    base_ok = sum(acc[base_method][k] for k in keys)
    method_ok = sum(acc[method][k] for k in keys)
    base_lost = sum(1 for k in keys if acc[base_method][k] == 1 and acc[method][k] == 0)
    method_won = sum(1 for k in keys if acc[base_method][k] == 0 and acc[method][k] == 1)
    print(
        f"  {label:<18} n={len(keys):4d}  "
        f"base {base_ok / len(keys) * 100:5.1f}%  "
        f"{method} {method_ok / len(keys) * 100:5.1f}%  "
        f"net {method_won - base_lost:+d} (win {method_won}/loss {base_lost})  "
        f"McNemar p={mcnemar_p(base_lost, method_won):.4g}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlb", default="/root/autodl-tmp/MMLB_subset")
    ap.add_argument("--eval", default="")
    ap.add_argument("--base", default="mmlb_base")
    ap.add_argument("--methods", default="mmlb_dg,mmlb_mm,mmlb_dg_mm")
    args = ap.parse_args()

    eval_path = args.eval or os.path.join(args.mmlb, "_eval", "llm_evaluation_results.json")
    acc = load_eval(eval_path)

    triggers = {
        "mmlb_dg": load_trigger_flags(args.mmlb, "qa_results_mmlb_dg.json", ["dg_used"]),
        "mmlb_mm": load_trigger_flags(args.mmlb, "qa_results_mmlb_mm.json", ["mm_ground_used"]),
        "mmlb_dg_mm": load_trigger_flags(
            args.mmlb,
            "qa_results_mmlb_dg_mm.json",
            ["dg_used", "mm_ground_used"],
        ),
    }

    print("==== MMLongBench-Doc: RAG-Anything variants ====")
    methods: List[str] = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in acc:
            print(f"\n{method}: no judged records")
            continue
        print(f"\n{method} vs {args.base}")
        summarize_pair(acc, args.base, method, "overall", lambda _k: True)
        flags = triggers.get(method, {})
        if flags:
            summarize_pair(acc, args.base, method, "triggered", lambda k, f=flags: f.get(k) is True)
            summarize_pair(acc, args.base, method, "not_triggered", lambda k, f=flags: f.get(k) is False)


if __name__ == "__main__":
    main()
