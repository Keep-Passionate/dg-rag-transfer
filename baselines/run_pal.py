"""基线③PAL / Program-of-Thoughts：LLM 每题现写 Python 在文档文本上计算，再执行。可续跑。

证明（最强概念对手）：就算让 LLM 自己写代码算，我们固定确定性算子也不输——且零 LLM、可复现。
用法：python run_pal.py --root /root/autodl-tmp/DocBench_subset [--limit N]
"""
import argparse
import os
import re

from base_common import doc_text, iter_docs, llm, rec, result_path, write_results

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/root/autodl-tmp/DocBench_subset")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--name", default="pal")
ap.add_argument("--max-chars", type=int, default=int(os.getenv("PAL_MAX_CHARS", "48000")))
ap.add_argument("--meta-only", type=int, default=1)
a = ap.parse_args()

_PROMPT = """You are given the full text of a document in a Python variable `text` (a string).
Write a Python function `solve(text)` that computes the answer to the question below and RETURNS it
as a short string (a number, name, date, or page). Use only the standard library and the `re` module.
Do not print; just return. Output ONLY a code block, no explanation.

Question: {q}

```python
def solve(text):
    # your code
    return answer
```"""

_SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                  for k in ("len", "range", "min", "max", "sum", "sorted", "set", "list", "dict",
                            "tuple", "str", "int", "float", "bool", "enumerate", "abs", "round",
                            "any", "all", "zip", "map", "filter", "reversed", "print")}


def _extract_code(txt):
    m = re.search(r"```(?:python)?\s*(.+?)```", txt, re.S)
    return (m.group(1) if m else txt).strip()


def _run(code, text):
    ns = {"re": re, "__builtins__": _SAFE_BUILTINS}
    exec(code, ns)                       # PAL 本质=执行生成代码；受限 builtins + 仅本机研究用
    if "solve" not in ns:
        return None
    return ns["solve"](text)


for folder, pdf, qs in iter_docs(a.root, a.limit, bool(a.meta_only)):
    if result_path(folder, a.name).exists():
        continue
    text = doc_text(pdf)[:a.max_chars]
    if not text:
        continue
    records = []
    for o in qs:
        q = o.get("question", "")
        ans = ""
        try:
            code = _extract_code(llm(_PROMPT.format(q=q)))
            val = _run(code, text)
            ans = "" if val is None else str(val).strip()
        except Exception as e:
            ans = ""                     # 代码跑挂=PAL 的真实失败模式，记空（不回退，公平对比）
            print(f"[{folder.name}] PAL 代码失败: {str(e)[:60]}")
        records.append(rec(o, ans))
    write_results(folder, a.name, records)
    print(f"[{folder.name}] {a.name} 完成 {len(records)} 题")
print("done:", a.name)
