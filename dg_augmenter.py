"""Backbone-agnostic DG/MM question augmenter for transfer experiments.

The transfer runners call ``augment(question, pdf_path)`` before handing the
question to GraphRAG, vanilla RAG, or any other backbone. By default this keeps
the original DG-RAG behavior: deterministic ``dg_core.ground`` evidence is
appended when it fires, otherwise the question is returned unchanged.

Set ``ENABLE_MM_GROUND=true`` to also append the portable multimodal grounding
evidence from ``mm_grounding``. That layer reads MinerU/RAG-Anything
``content_list`` JSON, but it has no dependency on RAG-Anything internals.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _core_dir() -> str:
    candidates = [
        os.getenv("DG_CORE_DIR", ""),
        os.getenv("MM_CORE_DIR", ""),
        "/root/autodl-tmp/rag-L1/reproduce",
        r"D:/project/RAG-Anything-main/reproduce",
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "dg_core.py")):
            return path
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "mm_grounding.py")):
            return path
    return candidates[-1]


DG_CORE_DIR = _core_dir()
if DG_CORE_DIR and DG_CORE_DIR not in sys.path:
    sys.path.insert(0, DG_CORE_DIR)

import dg_core  # noqa: E402

DG_GATE = os.getenv("DG_GATE", "all_fire")  # all_fire | self_check
_TAU = getattr(dg_core, "_THRESHOLD", {})


def _gate_pass(fact: Any) -> bool:
    if DG_GATE != "self_check":
        return True
    return float(getattr(fact, "confidence", 0.0)) >= _TAU.get(getattr(fact, "kind", ""), 0.6)


def _append_note(question: str, note: str) -> str:
    return f"{question}\n\n{note}" if note else question


def augment(
    question: str,
    pdf_path: str,
    content_list_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(augmented_question, meta)``.

    If all gates abstain, the augmented question is byte-for-byte identical to
    the input, so a transfer backbone can reuse its base answer.
    """

    augmented = question
    meta: Dict[str, Any] = {
        "dg_used": False,
        "dg_kind": "",
        "mm_ground_used": False,
        "mm_ground_kind": "",
        "mm_ground_vlm": False,
    }

    try:
        fact = dg_core.ground(question, pdf_path, content_list_path)
    except Exception:
        fact = None

    if fact and getattr(fact, "note", "") and _gate_pass(fact):
        augmented = _append_note(augmented, fact.note)
        meta.update(
            {
                "dg_used": True,
                "dg_kind": getattr(fact, "kind", ""),
                "dg_conf": round(float(getattr(fact, "confidence", 0.0)), 3),
            }
        )

    if _truthy("ENABLE_MM_GROUND"):
        try:
            import mm_augmenter  # local transfer adapter; imports mm_grounding lazily

            mm_augmented, mm_meta = mm_augmenter.augment(question, pdf_path, content_list_path)
            if mm_augmented != question:
                mm_note = mm_augmented[len(question) :].lstrip()
                augmented = _append_note(augmented, mm_note)
            meta.update(mm_meta)
        except Exception as exc:
            meta.update({"mm_ground_error": str(exc)[:160]})

    return augmented, meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the portable DG/MM augmenter.")
    ap.add_argument("pdf")
    ap.add_argument("question")
    ap.add_argument("--content-list")
    args = ap.parse_args()
    q, info = augment(args.question, args.pdf, args.content_list)
    print("meta:", info)
    print("---")
    print(q)
