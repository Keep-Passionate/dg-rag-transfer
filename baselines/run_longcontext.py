"""基线①Long-context（无检索）：整篇全文直灌 LLM。可续跑。

证明：把全文都给模型，全局属性题照样答不好——盲区不是检索覆盖率问题。
用法：python run_longcontext.py --root /root/autodl-tmp/DocBench_subset [--limit N]
"""
import argparse
import os

from base_common import RESP_HINT, doc_text, iter_docs, llm, rec, result_path, write_results

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/root/autodl-tmp/DocBench_subset")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--name", default="longcontext")
ap.add_argument("--max-chars", type=int, default=int(os.getenv("LC_MAX_CHARS", "48000")))
ap.add_argument("--meta-only", type=int, default=1)
a = ap.parse_args()

for folder, pdf, qs in iter_docs(a.root, a.limit, bool(a.meta_only)):
    if result_path(folder, a.name).exists():          # 续跑：已做过跳过
        continue
    text = doc_text(pdf)[:a.max_chars]
    if not text:
        continue
    records = []
    for o in qs:
        q = o.get("question", "")
        prompt = (f"Read the following document and answer the question.\n\n"
                  f"Document:\n{text}\n\nQuestion: {q}\n\n{RESP_HINT}\nAnswer:")
        try:
            ans = llm(prompt)
        except Exception as e:
            ans = ""
            print(f"[{folder.name}] LLM 失败，记空: {e}")
        records.append(rec(o, ans))
    write_results(folder, a.name, records)
    print(f"[{folder.name}] {a.name} 完成 {len(records)} 题")
print("done:", a.name)
