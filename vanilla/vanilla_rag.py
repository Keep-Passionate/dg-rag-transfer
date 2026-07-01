"""最朴素 native/vanilla RAG 骨干（Lewis 2020 形态）：嵌入切块 → 稠密 top-k 检索 → qwen 生成。

无图、无 rerank、无多模态、**无 LLM 建索引**（只嵌入，近乎免费）。作为 DG 骨干无关迁移的第三个骨干，
与 GraphRAG 版共用 dg_augmenter / 评测器 / compare_graphrag。模型走百炼 OpenAI 兼容端点，key 从 env 读。
"""
import json
import os
from pathlib import Path

import numpy as np

DASHSCOPE_BASE = os.getenv("DASHSCOPE_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CHAT_MODEL = os.getenv("VANILLA_CHAT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("VANILLA_EMBED_MODEL", "text-embedding-v3")
EMBED_BATCH = 10          # 百炼 text-embedding-v3 兼容端点单批 contents 上限 10
CHUNK_CHARS = int(os.getenv("VANILLA_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("VANILLA_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("VANILLA_TOP_K", "8"))
RESPONSE_HINT = ("Answer with a single short sentence; if the answer is a name, number, "
                 "date, or page, give only that value.")

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("GRAPHRAG_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY 未设置")
        _client = OpenAI(api_key=key, base_url=DASHSCOPE_BASE)
    return _client


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """字符级切块，尽量在空白处断开，重叠 overlap。零依赖、够用。"""
    text = text or ""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            sp = text.rfind(" ", i + size - overlap, end)
            if sp > i:
                end = sp
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


def embed(texts):
    c = client()
    out = []
    for s in range(0, len(texts), EMBED_BATCH):
        resp = c.embeddings.create(model=EMBED_MODEL, input=texts[s:s + EMBED_BATCH])
        out.extend(d.embedding for d in resp.data)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)          # 归一化 → cosine=点积


def build_index(text, out_dir):
    chunks = chunk_text(text)
    if not chunks:
        return 0
    emb = embed(chunks)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "emb.npy", emb)
    (out_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return len(chunks)


def index_ready(out_dir):
    out_dir = Path(out_dir)
    return (out_dir / "emb.npy").exists() and (out_dir / "chunks.json").exists()


def load_index(out_dir):
    out_dir = Path(out_dir)
    emb = np.load(out_dir / "emb.npy")
    chunks = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
    return emb, chunks


def retrieve(question, emb, chunks, k=TOP_K):
    q = embed([question])[0]
    sims = emb @ q
    idx = np.argsort(-sims)[:k]
    return [chunks[i] for i in idx]


def generate(question, ctx_chunks):
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(ctx_chunks))
    prompt = ("Use the following retrieved context to answer the question. If the context "
              "does not contain the answer, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\n\n{RESPONSE_HINT}\nAnswer:")
    resp = client().chat.completions.create(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return (resp.choices[0].message.content or "").strip()


def answer(question, emb, chunks, k=TOP_K):
    return generate(question, retrieve(question, emb, chunks, k))
