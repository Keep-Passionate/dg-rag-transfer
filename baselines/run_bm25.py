"""基线②BM25 稀疏检索 RAG：切块→BM25 词法 top-k→生成。可续跑。

证明：把稠密检索换成词法检索，全局属性盲区依然在——不是某种 embedding 的偶然缺陷。
依赖：pip install rank_bm25。用法：python run_bm25.py --root /root/autodl-tmp/DocBench_subset [--limit N]
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 复用 vanilla 的切块
from vanilla.vanilla_rag import chunk_text                        # noqa: E402
from base_common import RESP_HINT, doc_text, iter_docs, llm, rec, result_path, write_results  # noqa: E402

from rank_bm25 import BM25Okapi                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/root/autodl-tmp/DocBench_subset")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--name", default="bm25")
ap.add_argument("--top-k", type=int, default=int(os.getenv("BM25_TOP_K", "8")))
ap.add_argument("--meta-only", type=int, default=1)
a = ap.parse_args()

_tok = lambda s: re.findall(r"[a-z0-9]+", (s or "").lower())

for folder, pdf, qs in iter_docs(a.root, a.limit, bool(a.meta_only)):
    if result_path(folder, a.name).exists():
        continue
    chunks = chunk_text(doc_text(pdf))
    if not chunks:
        continue
    bm25 = BM25Okapi([_tok(c) for c in chunks])
    records = []
    for o in qs:
        q = o.get("question", "")
        scores = bm25.get_scores(_tok(q))
        top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:a.top_k]
        context = "\n\n".join(f"[{j + 1}] {chunks[i]}" for j, i in enumerate(top))
        prompt = ("Use the following retrieved context to answer the question. If the context "
                  "does not contain the answer, say you don't know.\n\n"
                  f"Context:\n{context}\n\nQuestion: {q}\n\n{RESP_HINT}\nAnswer:")
        try:
            ans = llm(prompt)
        except Exception as e:
            ans = ""
            print(f"[{folder.name}] LLM 失败，记空: {e}")
        records.append(rec(o, ans))
    write_results(folder, a.name, records)
    print(f"[{folder.name}] {a.name} 完成 {len(records)} 题（{len(chunks)} 块）")
print("done:", a.name)
