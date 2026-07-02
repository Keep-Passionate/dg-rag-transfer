"""Backbone-agnostic question augmenter for the DG-RAG transfer experiments.

Wraps dg_core.ground: for a question + its source PDF, compute the deterministic
verified fact (zero LLM, self-check gate A) and prepend it to the question. Any RAG
backbone then answers the *augmented* question unchanged. This is exactly how DG-RAG
attaches to RAG-Anything; reusing it in front of GraphRAG / a vanilla chunk-RAG /
other backbones tests the backbone-agnostic transfer claim.

Gate（由环境变量 DG_GATE 控制，须与论文口径一致）:
  - all_fire（默认，保持既有 GraphRAG 全量跑的行为不变）: dg_core.ground 有 Fact 就注入；
    论文口径的门控 A 需事后用 gate_ablation 式零 LLM mux 折算。
  - self_check: 在线门控 A —— 仅当 Fact 的实例自检分 >= dg_core._THRESHOLD[kind]
    （与主实验 gate_ablation 的 self_check(tau) 同一套固定阈值）才注入，否则弃权。
    弃权时返回原问题（恒等回退 = 骨干不受损），还能省一次 dg 查询。
"""
import os
import sys

# dg_core lives in the main thesis repo; override with DG_CORE_DIR if moved/on server.
DG_CORE_DIR = os.getenv("DG_CORE_DIR", r"D:/project/RAG-Anything-main/reproduce")
if DG_CORE_DIR not in sys.path:
    sys.path.insert(0, DG_CORE_DIR)
import dg_core  # noqa: E402

DG_GATE = os.getenv("DG_GATE", "all_fire")      # all_fire | self_check(=论文门控A)
_TAU = getattr(dg_core, "_THRESHOLD", {})       # 门控A逐类固定阈值（与 gate_ablation 同源）


def _gate_pass(fact):
    if DG_GATE != "self_check":
        return True
    return fact.confidence >= _TAU.get(fact.kind, 0.6)


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
    if fact and fact.note and _gate_pass(fact):
        return f"{question}\n\n{fact.note}", {"dg_used": True, "dg_kind": fact.kind,
                                              "dg_conf": round(float(fact.confidence), 3)}
    return question, {"dg_used": False, "dg_kind": ""}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test the DG augmenter on one question.")
    ap.add_argument("pdf")
    ap.add_argument("question")
    a = ap.parse_args()
    q, meta = augment(a.question, a.pdf)
    print("dg_used:", meta, "\n---\n", q)
