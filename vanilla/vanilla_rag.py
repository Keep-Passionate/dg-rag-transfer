"""最朴素 native/vanilla RAG 骨干（Lewis 2020 形态）：嵌入切块 → 稠密 top-k 检索 → qwen 生成。

无图、无 rerank、无多模态、**无 LLM 建索引**（只嵌入，近乎免费）。作为 DG 骨干无关迁移的第三个骨干，
与 GraphRAG 版共用 dg_augmenter / 评测器 / compare_graphrag。模型走百炼 OpenAI 兼容端点，key 从 env 读。

切块尺度：~600 token/块、重叠 100（token 级，tiktoken cl100k_base；缺 tiktoken 时按字符近似回退）。
"""
import json
import os
import time
from pathlib import Path

import numpy as np

DASHSCOPE_BASE = os.getenv("DASHSCOPE_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CHAT_MODEL = os.getenv("VANILLA_CHAT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("VANILLA_EMBED_MODEL", "text-embedding-v3")
EMBED_BATCH = 10          # 百炼 text-embedding-v3 兼容端点单批 contents 上限 10
CHUNK_TOKENS = int(os.getenv("VANILLA_CHUNK_TOKENS", "600"))     # ~600 token/块
CHUNK_OVERLAP = int(os.getenv("VANILLA_CHUNK_OVERLAP", "100"))   # 重叠 100 token
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


def _retry(fn, tries=4, base=2.0):
    """轻量指数退避：抗 429 / 偶发超时。最后一次仍失败则抛出（让上层决定跳过/记空）。"""
    for t in range(tries):
        try:
            return fn()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(base * (2 ** t))


# ---- token 级切块（tiktoken 优先；缺了按字符近似回退，仍尽量在空白处断开）----
_enc = None


def _encoder():
    global _enc
    if _enc is None:
        try:
            import tiktoken
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc = False          # 标记不可用 → 走字符回退
    return _enc


def chunk_text(text, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """滑窗切块：~chunk_tokens token/块、overlap 重叠。返回非空块列表（保持阅读顺序）。"""
    text = (text or "").strip()
    if not text:
        return []
    enc = _encoder()
    step = max(1, chunk_tokens - overlap)
    out = []
    if enc:
        toks = enc.encode(text)
        for i in range(0, len(toks), step):
            piece = toks[i:i + chunk_tokens]
            if not piece:
                break
            s = enc.decode(piece).strip()
            if s:
                out.append(s)
            if i + chunk_tokens >= len(toks):     # 已到末尾，别再补重复的小尾巴
                break
        return out
    # 字符回退：~4 char/token 粗估，尽量在空白处断开
    size = chunk_tokens * 4
    cover = overlap * 4
    cstep = max(1, size - cover)
    i, n = 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            sp = text.rfind(" ", i + size - cover, end)
            if sp > i:
                end = sp
        piece = text[i:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        i = max(end - cover, i + 1)
    return out


def embed(texts):
    c = client()
    out = []
    for s in range(0, len(texts), EMBED_BATCH):
        batch = texts[s:s + EMBED_BATCH]
        resp = _retry(lambda b=batch: c.embeddings.create(model=EMBED_MODEL, input=b))
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
    idx = np.argsort(-sims)[:max(1, k)]
    return [chunks[i] for i in idx]


def generate(question, ctx_chunks):
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(ctx_chunks))
    prompt = ("Use the following retrieved context to answer the question. If the context "
              "does not contain the answer, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\n\n{RESPONSE_HINT}\nAnswer:")
    resp = _retry(lambda: client().chat.completions.create(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0))
    return (resp.choices[0].message.content or "").strip()


def answer(question, emb, chunks, k=TOP_K):
    return generate(question, retrieve(question, emb, chunks, k))
