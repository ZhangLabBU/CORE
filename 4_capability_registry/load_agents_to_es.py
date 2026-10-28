import os, json
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import dashscope

# 显式指定 .env 文件路径
ENV_PATH = "/home/yzt/AgentDNS/src/react/.env"
load_dotenv(dotenv_path=ENV_PATH)

# 获取 .env 中的环境变量
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 其他环境变量
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_CA_CERT = os.getenv("ES_CA_CERT", "http_ca.crt")

# 打印检查环境变量是否正确加载
print(f"DASHSCOPE_API_KEY: {dashscope.api_key}")
print(f"ES_URL: {ES_URL}")

INDEX = os.getenv("CAP_INDEX", "capability_registry")
SEED_MD = os.getenv("SEED_MD", r"/home/yzt/AgentDNS/src/Pre_processing/k-v.md")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v2")


def es_client() -> Elasticsearch:
    if not ES_PASSWORD:
        raise RuntimeError("ES_PASSWORD 未设置，请在 .env 里配置 ES_PASSWORD")
    return Elasticsearch(
        ES_URL,
        basic_auth=(ES_USER, ES_PASSWORD),
    )


def clean_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
    return s.strip()


def parse_seeds(md_text: str):
    """
    一条 Value(JSON) => 一条 capability 文档。
    Key 行（含 ||）作为 keys[]。
    """
    lines = md_text.splitlines()
    docs = []
    i = 0
    while i < len(lines):
        if "{" in lines[i]:
            buf, brace, started = [], 0, False
            j = i
            while j < len(lines):
                line = lines[j]
                if "{" in line:
                    brace += line.count("{")
                    started = True
                if started:
                    buf.append(line)
                if "}" in line:
                    brace -= line.count("}")
                    if started and brace <= 0:
                        break
                j += 1

            if started and brace <= 0:
                json_text = clean_json_text("\n".join(buf))

                # 向上找最近的 keys 行
                k = i - 1
                key_line = ""
                while k >= 0 and (i - k) <= 30:
                    if "||" in lines[k]:
                        key_line = lines[k].strip()
                        break
                    k -= 1

                try:
                    obj = json.loads(json_text)
                except Exception:
                    i = j + 1
                    continue

                keys = [x.strip() for x in key_line.split("||") if x.strip()] if key_line else []
                obj["keys"] = keys

                # 兼容缺省字段
                obj.setdefault("example_queries", [])
                obj.setdefault("required_slots", [])
                obj.setdefault("optional_slots", [])
                obj.setdefault("tools_involved", [])
                obj.setdefault("relationship_type", "")

                # 统一 list 类型（避免 ES 映射报错）
                for kk in ["keys", "example_queries", "required_slots", "optional_slots", "tools_involved"]:
                    v = obj.get(kk, [])
                    if isinstance(v, str):
                        v = [v]
                    obj[kk] = v

                parts = [
                    obj.get("capability_id", ""),
                    obj.get("agent_name", ""),
                    obj.get("description", ""),
                    " ".join(obj.get("keys", [])),
                    " ".join(obj.get("example_queries", [])),
                    "required: " + " ".join(obj.get("required_slots", [])),
                    "optional: " + " ".join(obj.get("optional_slots", [])),
                    "tools: " + " ".join(obj.get("tools_involved", [])),
                    obj.get("fallback_prompt", ""),
                    obj.get("relationship_type", ""),
                ]
                obj["search_text"] = " | ".join([p for p in parts if p])
                docs.append(obj)

                i = j + 1
                continue
        i += 1
    return docs


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = dashscope.TextEmbedding.call(model=EMBED_MODEL, input=texts)
    if resp.get("status_code") != 200:
        raise RuntimeError(f"Embedding failed: {resp}")
    out = resp.get("output", {})
    if "embeddings" in out:
        return [it["embedding"] for it in out["embeddings"]]
    if "data" in out:
        return [it["embedding"] for it in out["data"]]
    raise RuntimeError(f"Unknown embedding response format: {resp}")


def ensure_index(es: Elasticsearch, dims: int):
    if es.indices.exists(index=INDEX):
        return

    mappings = {
        "properties": {
            "capability_id": {"type": "keyword"},
            "agent_name": {"type": "keyword"},
            "url": {"type": "keyword"},
            "description": {"type": "text"},
            "keys": {"type": "text"},
            "example_queries": {"type": "text"},
            "required_slots": {"type": "keyword"},
            "optional_slots": {"type": "keyword"},
            "fallback_prompt": {"type": "text"},
            "relationship_type": {"type": "keyword"},
            "tools_involved": {"type": "keyword"},
            "search_text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
        }
    }
    es.indices.create(index=INDEX, mappings=mappings)


def main():
    if not dashscope.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置（.env 中配置）")

    md_text = Path(SEED_MD).read_text(encoding="utf-8")
    docs = parse_seeds(md_text)
    print("Parsed capability docs:", len(docs))
    if not docs:
        raise RuntimeError("没有从 k-v.md 解析出任何 capability JSON，请检查格式。")

    es = es_client()

    one_emb = embed_texts([docs[0]["search_text"]])[0]
    dims = len(one_emb)
    print("Detected embedding dims:", dims)

    ensure_index(es, dims=dims)

    embs = [one_emb] + embed_texts([d["search_text"] for d in docs[1:]])

    actions = []
    for d, e in zip(docs, embs):
        d["embedding"] = e
        doc_id = d.get("capability_id") or d.get("agent_name")
        actions.append({"_index": INDEX, "_id": doc_id, "_source": d})

    ok, errors = bulk(es, actions, raise_on_error=False)
    es.indices.refresh(index=INDEX)
    print("Indexed:", ok, "errors:", len(errors))
    print("Index:", INDEX)


if __name__ == "__main__":
    main()
