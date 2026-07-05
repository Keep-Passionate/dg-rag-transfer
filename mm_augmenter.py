"""Portable multimodal grounding adapter for transfer backbones.

This module is intentionally tiny: the real operator lives in
``mm_grounding.py`` in the main thesis repo's ``reproduce`` directory. The
adapter keeps transfer experiments independent of RAG-Anything query code while
still reusing the exact same content-list grounding logic.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple


def _candidate_dirs():
    yield os.getenv("MM_CORE_DIR", "")
    yield os.getenv("DG_CORE_DIR", "")
    yield "/root/autodl-tmp/rag-L1/reproduce"
    yield r"D:/project/RAG-Anything-main/reproduce"


def _find_core_dir() -> str:
    for path in _candidate_dirs():
        if path and os.path.exists(os.path.join(path, "mm_grounding.py")):
            return path
    raise RuntimeError("mm_grounding.py not found; set MM_CORE_DIR or DG_CORE_DIR to the main repo reproduce dir")


MM_CORE_DIR = _find_core_dir()
if MM_CORE_DIR not in sys.path:
    sys.path.insert(0, MM_CORE_DIR)

import mm_grounding  # noqa: E402


def augment(
    question: str,
    pdf_path: str,
    content_list_path: Optional[str] = None,
    model: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(augmented_question, meta)`` for any text-taking backbone.

    Image evidence is represented as ``Image Path: ...`` lines in the appended
    note. A text-only backbone will at least see captions, table bodies, page
    anchors, and neighboring OCR text; a VLM-capable backbone can additionally
    consume the image paths from ``meta["mm_ground_evidence"]``.
    """

    fact = mm_grounding.ground(question, pdf_path, content_list_path, model=model)
    if not fact or not getattr(fact, "note", ""):
        return question, {"mm_ground_used": False, "mm_ground_kind": "", "mm_ground_vlm": False}

    evidence = []
    for item in getattr(fact, "evidence", []) or []:
        evidence.append(
            {
                "element_id": getattr(item, "element_id", ""),
                "kind": getattr(item, "kind", ""),
                "label": getattr(item, "label", ""),
                "page": getattr(item, "page", None),
                "image_path": getattr(item, "image_path", ""),
            }
        )

    meta: Dict[str, Any] = {
        "mm_ground_used": True,
        "mm_ground_kind": getattr(fact, "kind", ""),
        "mm_ground_conf": round(float(getattr(fact, "confidence", 0.0)), 3),
        "mm_ground_vlm": bool(getattr(fact, "requires_vlm", False)),
        "mm_ground_evidence": evidence,
    }
    return f"{question}\n\n{fact.note}", meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test multimodal grounding on one question.")
    ap.add_argument("pdf")
    ap.add_argument("question")
    ap.add_argument("--content-list")
    args = ap.parse_args()
    q, info = augment(args.question, args.pdf, args.content_list)
    print("meta:", info)
    print("---")
    print(q)
