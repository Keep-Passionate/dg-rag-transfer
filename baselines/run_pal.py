"""基线③PAL / Program-of-Thoughts：LLM 每题现写 Python 在文档文本上计算，再执行。可续跑。

证明（最强概念对手）：就算让 LLM 自己写代码算，我们固定确定性算子也不输——且零 LLM、可复现。
用法：python run_pal.py --root /root/autodl-tmp/DocBench_subset [--limit N]
"""
import argparse
import os
import re
import signal

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
as a short string (a number, name, date, or page). You may use the modules re, collections, math,
string, itertools, statistics, functools. Do not print; just return. Output ONLY a code block, no
explanation.

Question: {q}

```python
def solve(text):
    # your code
    return answer
```"""

_SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                  for k in ("len", "range", "min", "max", "sum", "sorted", "set", "list", "dict",
                            "tuple", "str", "int", "float", "bool", "enumerate", "abs", "round",
                            "any", "all", "zip", "map", "filter", "reversed", "print", "isinstance",
                            "type", "chr", "ord", "repr", "slice", "next", "iter", "frozenset")}

# 允许 LLM 生成的代码 import 标准库中安全的几个模块（否则 `import re/collections` 报
# "__import__ not found"，PAL 几乎全挂）。仅白名单，纯计算用。
_ALLOWED_MODS = {"re", "collections", "math", "string", "itertools", "statistics", "functools"}


def _safe_import(name, *a, **k):
    if name.split(".")[0] in _ALLOWED_MODS:
        return __import__(name, *a, **k)
    raise ImportError(f"PAL 沙箱不允许 import '{name}'")


_SAFE_BUILTINS["__import__"] = _safe_import


def _extract_code(txt):
    m = re.search(r"```(?:python)?\s*(.+?)```", txt, re.S)
    return (m.group(1) if m else txt).strip()


class _Timeout(Exception):
    pass


def _run(code, text, timeout=10):
    # PAL 本质=执行生成代码；受限 builtins + signal 超时（防 LLM 生成死循环把整个跑挂住）。
    import collections
    import math
    import string
    ns = {"re": re, "collections": collections, "math": math, "string": string,
          "__builtins__": _SAFE_BUILTINS}
    has_alarm = hasattr(signal, "SIGALRM")          # Linux 有；Windows 没有则跳过超时
    if has_alarm:
        old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
        signal.alarm(timeout)
    try:
        exec(code, ns)                   # 仅本机研究用
        return ns["solve"](text) if "solve" in ns else None
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


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
