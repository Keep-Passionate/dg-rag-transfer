"""把 `graphrag init` 生成的 settings.yaml 就地改成 qwen-plus / text-embedding-v3 / 百炼端点。

为什么不手写整份：GraphRAG 各版本 schema 不同。改法=先 `graphrag init` 生成**与实装版本匹配**
的默认配置，本脚本只覆盖模型名/端点/密钥，不动其它默认结构。

GraphRAG 3.x（服务器实装 3.1.0）schema：
  completion_models: { default_completion_model: {model_provider, model, auth_method, api_key, retry} }
  embedding_models:  { default_embedding_model:  {model_provider, model, auth_method, api_key, retry} }
默认**没有 api_base**（指向 OpenAI）——本脚本注入 `api_base` 指到百炼 OpenAI 兼容端点。
api_key 用 ${GRAPHRAG_API_KEY}（从环境变量/.env 读，绝不写死密钥）。
兼容旧 2.x 的 models: 块（fallback）。

用法：python patch_settings.py <root>/settings.yaml
"""
import os
import sys

import yaml

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT = "qwen-plus"
EMBED = "text-embedding-v3"


def patch_model(m, is_embed):
    if not isinstance(m, dict):
        return
    m["model_provider"] = "openai"
    m["model"] = EMBED if is_embed else CHAT
    m["auth_method"] = m.get("auth_method", "api_key")
    m["api_key"] = "${GRAPHRAG_API_KEY}"
    m["api_base"] = BASE


def main(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    touched = False
    if "completion_models" in cfg or "embedding_models" in cfg:          # 3.x
        for _, m in (cfg.get("completion_models") or {}).items():
            patch_model(m, False)
            touched = True
        for _, m in (cfg.get("embedding_models") or {}).items():
            patch_model(m, True)
            touched = True
    elif isinstance(cfg.get("models"), dict):                            # 2.x fallback
        for name, m in cfg["models"].items():
            emb = "embed" in str(m.get("type", "")).lower() or "embed" in name.lower()
            patch_model(m, emb)
            touched = True

    if not touched:
        print("!! 未识别 settings schema（无 completion_models/embedding_models/models）——把 settings.yaml 贴回")
        sys.exit(2)

    # 百炼 text-embedding-v3 的 OpenAI 兼容端点：单批 contents 上限 10，超了报 400 InvalidParameter
    # ("batch size ... not larger than 10")→嵌入步失败、整篇索引挂。这里把嵌入批大小压到 10。
    et = cfg.get("embed_text")
    if not isinstance(et, dict):
        et = {}
        cfg["embed_text"] = et
    et["batch_size"] = 10

    # 可选：GRAPHRAG_CHUNK_SIZE 覆盖切块 token 数——治个别文档待嵌入文本超 text-embedding-v3
    # 8192 输入上限（如 doc 214 报 "Range of input length should be [1, 8192]"）。
    # 必须走这里而不是手改 settings.yaml：驱动重建时 init --force 会重新生成配置。
    cs = os.getenv("GRAPHRAG_CHUNK_SIZE")
    if cs:
        ch = cfg.get("chunking")
        if not isinstance(ch, dict):
            ch = {}
            cfg["chunking"] = ch
        ch["size"] = int(cs)
        ch["overlap"] = min(int(ch.get("overlap", 100) or 100), max(1, int(cs) // 6))
        print(f"chunking.size -> {cs} (GRAPHRAG_CHUNK_SIZE)")

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"patched: {path}  chat={CHAT} embed={EMBED} api_base={BASE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python patch_settings.py <settings.yaml>")
        sys.exit(1)
    main(sys.argv[1])
