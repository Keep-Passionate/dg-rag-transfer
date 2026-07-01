"""把 `graphrag init` 生成的 settings.yaml 就地改成 qwen-plus / text-embedding-v3 / 百炼端点。

为什么不直接手写整份 settings.yaml：GraphRAG 各版本 schema 有别（0.x 用 llm:/embeddings:，
1.x/2.x 用 models: 块）。改法=先 `graphrag init` 让它生成**与实装版本匹配**的默认配置，
再由本脚本只覆盖模型/端点/批大小几个字段——自适应版本，不动其它默认结构。

api_key 用 `${GRAPHRAG_API_KEY}`（GraphRAG 会从环境变量/`.env` 读，绝不写死密钥）。

用法：python patch_settings.py <root>/settings.yaml
"""
import sys

import yaml

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT = "qwen-plus"
EMBED = "text-embedding-v3"


def is_embed(name, m):
    t = str(m.get("type", "")).lower()
    return "embed" in t or "embed" in name.lower()


def main(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models")
    if not isinstance(models, dict):
        print("!! 未找到 models: 块（可能是 0.x schema）——请把版本贴回，我改用 llm:/embeddings: 写法。")
        sys.exit(2)

    for name, m in models.items():
        if not isinstance(m, dict):
            continue
        m["api_base"] = BASE
        m["api_key"] = "${GRAPHRAG_API_KEY}"
        m["concurrent_requests"] = 4
        if is_embed(name, m):
            m["model"] = EMBED
            m["type"] = m.get("type", "openai_embedding")
        else:
            m["model"] = CHAT
            m["type"] = m.get("type", "openai_chat")
            # qwen 的 OpenAI 兼容端点对 response_format=json 支持不稳，关掉走文本解析更保险
            m["model_supports_json"] = False

    # 嵌入批大小：百炼 text-embedding-v3 兼容端点单批上限较小，设 10 防 400/批超限
    et = cfg.get("embed_text")
    if isinstance(et, dict):
        et["batch_size"] = min(int(et.get("batch_size", 16) or 16), 10)

    # 输入统一为纯文本（build_input.py 产出的 .txt）
    inp = cfg.get("input")
    if isinstance(inp, dict):
        inp.setdefault("file_type", "text")

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"patched: {path}  chat={CHAT} embed={EMBED} base={BASE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python patch_settings.py <settings.yaml>")
        sys.exit(1)
    main(sys.argv[1])
