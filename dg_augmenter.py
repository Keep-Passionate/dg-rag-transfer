"""Backbone-agnostic question augmenter for the DG-RAG transfer experiments.

Wraps dg_core.ground: for a question + its source PDF, compute the deterministic
verified fact (zero LLM, self-check gate A) and prepend it to the question. Any RAG
backbone then answers the *augmented* question unchanged. This is exactly how DG-RAG
attaches to RAG-Anything; reusing it in front of GraphRAG / a vanilla chunk-RAG /
other backbones tests the backbone-agnostic transfer claim.

Gate: dg_core.ground returns a Fact only when the operator's self-check deems it
reliable; otherwise note is empty -> we return the question unchanged (identity
fallback = the backbone is never harmed).
"""
import os
import sys

# dg_core lives in the main thesis repo; override with DG_CORE_DIR if moved/on server.
DG_CORE_DIR = os.getenv("DG_CORE_DIR", r"D:/project/RAG-Anything-main/reproduce")
if DG_CORE_DIR not in sys.path:
    sys.path.insert(0, DG_CORE_DIR)
import dg_core  # noqa: E402


def augment(question: str, pdf_path: str, content_list_path: str = None):
    """Return (augmented_question, meta_dict).

    If the gate abstains, augmented_question == question (identity), so the backbone
    behaves exactly as without DG-RAG on that question.
    """
    fact = None
    try:
        fact = dg_core.ground(question, pdf_path, content_list_path)
    except Exception:
        fact = None
    if fact and fact.note:
        return f"{question}\n\n{fact.note}", {"dg_used": True, "dg_kind": fact.kind}
    return question, {"dg_used": False, "dg_kind": ""}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test the DG augmenter on one question.")
    ap.add_argument("pdf")
    ap.add_argument("question")
    a = ap.parse_args()
    q, meta = augment(a.question, a.pdf)
    print("dg_used:", meta, "\n---\n", q)
