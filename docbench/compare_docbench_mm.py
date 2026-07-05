"""Compare DocBench multimodal grounding variants after LLM judging."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from math import comb
from typing import Callable, Dict, Iterable, Tuple

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


def load_gold_types(docbench: str) -> Dict[Key, str]:
    gold: Dict[Key, str] = {}
    for folder in glob.glob(os.path.join(docbench, "*")):
        if not os.path.isdir(folder):
            continue
        pdf = next((os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.pdf"))), "")
        qa = os.path.join(folder, os.path.basename(folder), f"{os.path.basename(folder)}_qa.jsonl")
        if not os.path.exists(qa):
            qa = os.path.join(folder, f"{os.path.basename(folder)}_qa.jsonl")
        if not pdf or not os.path.exists(qa):
            continue
        for line in open(qa, encoding="utf-8"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            gold[(pdf, norm(row.get("question", "")))] = row.get("type", "")
    return gold


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
        out.setdefault(method, {})[(row.get("doc_id", ""), norm(row.get("question", "")))] = acc
    return out


def load_trigger_flags(docbench: str, filename: str, fields: Iterable[str]) -> Dict[Key, bool]:
    flags: Dict[Key, bool] = {}
    for path in glob.glob(os.path.join(docbench, "*", filename)):
        folder = os.path.dirname(path)
        pdf = next((os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.pdf"))), "")
        if not pdf:
            continue
        try:
            rows = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for row in rows:
            flags[(pdf, norm(row.get("question", "")))] = any(bool(row.get(field)) for field in fields)
    return flags


def print_accuracy_by_type(acc: Dict[str, Dict[Key, int]], gold: Dict[Key, str], methods: Iterable[str]) -> None:
    print("==== Accuracy by question type ====")
    for method in methods:
        if method not in acc:
            print(f"\n== {method}: no judged records ==")
            continue
        buckets = defaultdict(lambda: [0, 0])
        for key, value in acc[method].items():
            qtype = gold.get(key)
            if qtype is None:
                continue
            buckets[qtype][0] += value
            buckets[qtype][1] += 1
            buckets["__overall__"][0] += value
            buckets["__overall__"][1] += 1
        print(f"\n== {method} ==")
        for qtype in sorted(buckets):
            good, total = buckets[qtype]
            label = "overall" if qtype == "__overall__" else (qtype or "(no_type)")
            if total:
                print(f"  {label:<18} {good}/{total} = {good / total * 100:.1f}%")


def summarize_pair(
    acc: Dict[str, Dict[Key, int]],
    base: str,
    method: str,
    label: str,
    predicate: Callable[[Key], bool],
) -> None:
    keys = [k for k in acc.get(method, {}) if k in acc.get(base, {}) and predicate(k)]
    if not keys:
        print(f"  {label:<18} n=0")
        return
    base_ok = sum(acc[base][k] for k in keys)
    method_ok = sum(acc[method][k] for k in keys)
    base_lost = sum(1 for k in keys if acc[base][k] == 1 and acc[method][k] == 0)
    method_won = sum(1 for k in keys if acc[base][k] == 0 and acc[method][k] == 1)
    print(
        f"  {label:<18} n={len(keys):4d}  "
        f"{base} {base_ok / len(keys) * 100:5.1f}%  "
        f"{method} {method_ok / len(keys) * 100:5.1f}%  "
        f"net {method_won - base_lost:+d} (win {method_won}/loss {base_lost})  "
        f"McNemar p={mcnemar_p(base_lost, method_won):.4g}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docbench", default="/root/autodl-tmp/DocBench_subset")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--base", default="paperbase")
    ap.add_argument("--methods", default="mmground_docbench,dg_mm_docbench")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    acc = load_eval(args.eval)
    gold = load_gold_types(args.docbench)
    base = args.base
    if base not in acc and "mm_base_docbench" in acc:
        base = "mm_base_docbench"
        print(f"[base] {args.base!r} not judged here; using {base!r}")
    print_accuracy_by_type(acc, gold, [base] + methods)

    triggers = {
        "mmground_docbench": load_trigger_flags(args.docbench, "qa_results_mmground_docbench.json", ["mm_ground_used"]),
        "dg_mm_docbench": load_trigger_flags(
            args.docbench,
            "qa_results_dg_mm_docbench.json",
            ["dg_used", "mm_ground_used"],
        ),
    }

    print("\n==== Paired deltas vs base ====")
    for method in methods:
        if method not in acc:
            print(f"\n{method}: no judged records")
            continue
        print(f"\n{method} vs {base}")
        summarize_pair(acc, base, method, "overall", lambda _k: True)
        flags = triggers.get(method, {})
        if flags:
            summarize_pair(acc, base, method, "triggered", lambda k, f=flags: f.get(k) is True)
            summarize_pair(acc, base, method, "not_triggered", lambda k, f=flags: f.get(k) is False)


if __name__ == "__main__":
    main()
