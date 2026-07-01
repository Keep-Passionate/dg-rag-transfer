"""把 MinerU 解析出的 content_list 序列化成 GraphRAG 的输入 .txt（版本无关，零 LLM）。

迁移实验只动"骨干"一个变量。为公平，GraphRAG 与 RAG-Anything 吃**同一份解析文本**
（MinerU content_list），而不是各自重解析 PDF——既避免解析差异污染对比，又省一次解析钱。

序列化规则（保持阅读顺序）：
  header      -> 以 '#' 前缀成 Markdown 标题（text_level 决定层级）
  text/list   -> 原文
  table       -> table_body（MinerU 已给 HTML/Markdown 表体）
  image/chart -> "[Figure: <img_caption>]"（只留题注，不塞二进制）
  equation    -> text（LaTeX）
  page_number/page_footnote/bbox 等 -> 丢弃
content_list 缺失时回退 PyMuPDF 全文（需 fitz）。

用法：
  python build_input.py <pdf_path> -o <out.txt> [--content-list <cl.json>]
content_list 未显式给出时，按 PARSE_OUTPUT_DIR 用 dg_core.locate_content_list 自动定位
（服务器：PARSE_OUTPUT_DIR=/root/autodl-tmp/content_lists）。
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 复用 dg_core 的 content_list 定位逻辑（与接地层看同一份解析，口径一致）。
DG_CORE_DIR = os.getenv("DG_CORE_DIR", r"D:/project/RAG-Anything-main/reproduce")
if DG_CORE_DIR not in sys.path:
    sys.path.insert(0, DG_CORE_DIR)


def resolve_content_list(pdf_path: str, explicit: str = None) -> str | None:
    if explicit and os.path.exists(explicit):
        return explicit
    try:
        import dg_core  # 需要 fitz；仅在自动定位时才导入
        cl = dg_core.locate_content_list(pdf_path)
        if cl and os.path.exists(str(cl)):
            return str(cl)
    except Exception:
        pass
    return None


def serialize_content_list(items: list) -> str:
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        t = it.get("type", "")
        if t == "text":
            lvl = it.get("text_level")
            txt = (it.get("text") or "").strip()
            if not txt:
                continue
            out.append(("#" * int(lvl) + " " + txt) if lvl else txt)
        elif t in ("list", "code", "aside_text"):
            txt = (it.get("text") or "").strip()
            if txt:
                out.append(txt)
        elif t == "equation":
            txt = (it.get("text") or "").strip()
            if txt:
                out.append(txt)
        elif t == "table":
            body = (it.get("table_body") or "").strip()
            cap = " ".join(it.get("table_caption") or []).strip() if isinstance(
                it.get("table_caption"), list) else (it.get("table_caption") or "")
            if cap:
                out.append(f"Table: {cap}".strip())
            if body:
                out.append(body)
        elif t in ("image", "chart"):
            cap = it.get("img_caption")
            cap = " ".join(cap) if isinstance(cap, list) else (cap or "")
            cap = cap.strip()
            if cap:
                out.append(f"[Figure: {cap}]")
        # page_number / page_footnote / header(极少) / 其它 -> 丢弃
        elif t == "header":
            txt = (it.get("text") or "").strip()
            if txt:
                out.append("# " + txt)
    return "\n\n".join(out)


def pdf_fulltext(pdf_path: str) -> str:
    import fitz
    with fitz.open(pdf_path) as doc:
        return "\n\n".join(p.get_text() for p in doc)


def build_input(pdf_path: str, out_path: str, content_list: str = None) -> dict:
    cl = resolve_content_list(pdf_path, content_list)
    source = "content_list"
    text = ""
    if cl:
        with open(cl, encoding="utf-8") as f:
            items = [x for x in json.load(f) if isinstance(x, dict)]
        text = serialize_content_list(items)
    if not text.strip():
        source = "pymupdf"
        text = pdf_fulltext(pdf_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"source": source, "content_list": cl, "chars": len(text), "out": out_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="content_list -> GraphRAG input .txt")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--content-list", default=None)
    a = ap.parse_args()
    info = build_input(a.pdf, a.out, a.content_list)
    print(json.dumps(info, ensure_ascii=False))
