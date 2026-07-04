"""DocBench 对比基线共享工具：文档全文 / LLM 客户端（同主实验端点模型）/ 结果读写。

生成器与主实验一致（qwen-plus + LLM_BINDING_HOST），保证"唯一变量是怎么拿到答案"。
key/host 从环境变量读（run_baselines.sh 从主仓 .env 导出）。
"""
import json
import os
import time
from pathlib import Path

RESP_HINT = ("Answer with a single short sentence; if the answer is a name, number, "
             "date, or page, give only that value.")

_client = None


def _from_env_file(name):
    """env 里没有时，直接从主仓 .env 读（免去手动 export，nohup 也不怕丢环境）。"""
    cands = [os.environ.get("DG_ENV_FILE"), "/root/autodl-tmp/rag-L1/.env",
             os.path.join(os.getcwd(), ".env")]
    for p in cands:
        if p and os.path.exists(p):
            for ln in open(p, encoding="utf-8", errors="ignore"):
                ln = ln.strip()
                if ln.startswith(name + "="):
                    v = ln.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        return v
    return None


def _mk_client():
    from openai import OpenAI
    key = (os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
           or _from_env_file("LLM_BINDING_API_KEY"))
    host = (os.environ.get("LLM_BINDING_HOST") or _from_env_file("LLM_BINDING_HOST")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not key:
        raise RuntimeError("LLM_BINDING_API_KEY 未设置（env 与 .env 都没读到；"
                           "可设 DG_ENV_FILE 指向主仓 .env）")
    return OpenAI(api_key=key, base_url=host)


def llm(prompt, temperature=0, model=None):
    """一次对话补全，指数退避抗 429/超时。"""
    global _client
    if _client is None:
        _client = _mk_client()
    model = model or os.environ.get("BASELINE_CHAT_MODEL", "qwen-plus")
    for t in range(4):
        try:
            r = _client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if t == 3:
                raise
            time.sleep(2.0 * (2 ** t))


def doc_text(pdf_path):
    """PyMuPDF 取全文（与 dg_core 同源）。失败返回空串。"""
    try:
        import fitz
        with fitz.open(pdf_path) as d:
            return "\n".join(p.get_text() for p in d)
    except Exception:
        return ""


def iter_docs(root, limit=0, meta_only=True):
    """遍历 DocBench_subset/<id>/：yield (folder_path, pdf_path, questions[])。
    meta_only=True 只保留 type=='meta-data' 的题（对齐主表 + 省 LLM 钱）；无 meta 题的文档跳过。"""
    n = 0
    for f in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        qa = f / f"{f.name}_qa.jsonl"
        pdf = next((x for x in f.glob("*.pdf")), None)
        if not qa.exists() or pdf is None:
            continue
        qs = [json.loads(l) for l in open(qa, encoding="utf-8") if l.strip()]
        if meta_only:
            qs = [o for o in qs if o.get("type") == "meta-data"]
        if not qs:
            continue
        yield f, str(pdf), qs
        n += 1
        if limit and n >= limit:
            break


def result_path(folder, name):
    return Path(folder) / f"qa_results_{name}.json"


def write_results(folder, name, records):
    p = result_path(folder, name)
    json.dump(records, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return p


def rec(o, answer):
    """统一结果记录格式（对齐主仓评测器：question/answer/correct_answer）。"""
    return {"question": o.get("question", ""), "answer": answer,
            "correct_answer": o.get("answer", ""), "type": o.get("type", "")}
