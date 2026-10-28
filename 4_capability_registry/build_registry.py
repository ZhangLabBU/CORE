import os
import json
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import dashscope


# ----------------------------
# Path helpers
# ----------------------------
THIS_DIR = Path(__file__).resolve().parent  # .../AgentDNS/src/Pre_processing

def pick_existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None

def resolve_maybe_relative(p: str | None, bases: list[Path]) -> str | None:
    """Resolve a path that may be relative; try multiple bases."""
    if not p:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    for base in bases:
        cand = (base / pp).resolve()
        if cand.exists():
            return str(cand)
    # 如果都不存在，也给一个“最可能”的路径，方便你 debug
    return str((bases[0] / pp).resolve())


# ----------------------------
# Locate project root and .env
# ----------------------------
# 常见情况：.../AgentDNS/src/Pre_processing -> project root = .../AgentDNS
CANDIDATE_ROOTS = [
    THIS_DIR.parents[2],        # .../AgentDNS
    THIS_DIR.parents[3],        # 备用
    Path.cwd(),                 # 备用
]

PROJECT_ROOT = CANDIDATE_ROOTS[0]

# 你原来写在 src/react/.env（但你的 PROJECT_ROOT 可能会重复拼 AgentDNS）
env_candidates = [
    PROJECT_ROOT / "src" / "react" / ".env",
    PROJECT_ROOT / "AgentDNS" / "src" / "react" / ".env",
    THIS_DIR / ".env",
]
ENV_PATH = pick_existing_path(env_candidates)

if ENV_PATH is None:
    raise RuntimeError(
        f"找不到 .env 文件，尝试过：\n" + "\n".join(str(p) for p in env_candidates)
    )

load_dotenv(dotenv_path=ENV_PATH)


# ----------------------------
# Read env
# ----------------------------
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200").strip()
ES_USER = (os.getenv("ES_USER") or "").strip()
ES_PASSWORD = (os.getenv("ES_PASSWORD") or "").strip()
ES_CA_CERT = (os.getenv("ES_CA_CERT") or "").strip()

INDEX = os.getenv("ES_INDEX", "capability_registry").strip()

SEED_MD = (os.getenv("SEED_MD") or "./k-v.md").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v2").strip()
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1024"))

# 输出时不要打印真实 key
def mask(s: str | None) -> str:
    if not s:
        return "(empty)"
    if len(s) <= 8:
        return "****"
    return s[:3] + "****" + s[-4:]

print("Loaded .env from:", ENV_PATH)
print("DASHSCOPE_API_KEY:", mask(dashscope.api_key))
print("ES_URL:", ES_URL)

# 路径解析：同时尝试以 PROJECT_ROOT 和 THIS_DIR 作为 base
BASES = [PROJECT_ROOT, THIS_DIR, Path.cwd()]

SEED_MD_ABS = resolve_maybe_relative(SEED_MD, BASES)
# 注意：HTTP 模式下 ES_CA_CERT 可以为空；不要强行 resolve
ES_CA_CERT_ABS = resolve_maybe_relative(ES_CA_CERT, BASES) if ES_CA_CERT else None

print("SEED_MD path:", SEED_MD_ABS)
print("ES_CA_CERT path:", ES_CA_CERT_ABS or "(not used / empty)")


# ----------------------------
# Elasticsearch client
# ----------------------------
def es_client() -> Elasticsearch:
    """
    - http://  => 不要传 ca_certs（否则会触发 TLS 校验报错）
    - https:// => 才传 ca_certs
    - 是否传 basic_auth：只有当 ES_USER/ES_PASSWORD 都不为空时才传
    """
    u = urlparse(ES_URL)
    use_auth = bool(ES_USER and ES_PASSWORD)

    if u.scheme == "http":
        if use_auth:
            return Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD))
        return Elasticsearch(ES_URL)

    # https
    kwargs = {}
    if use_auth:
        kwargs["basic_auth"] = (ES_USER, ES_PASSWORD)
    if ES_CA_CERT_ABS:
        kwargs["ca_certs"] = ES_CA_CERT_ABS
    return Elasticsearch(ES_URL, **kwargs)


# ----------------------------
# Index creation (ES 7 vs 8 compatible)
# ----------------------------
def ensure_index(es: Elasticsearch, dims: int):
    if es.indices.exists(index=INDEX):
        return

    info = es.info()
    version_str = info.get("version", {}).get("number", "7.0.0")
    major = int(version_str.split(".")[0])

    # 7.x：dense_vector 通常只支持 dims（用 script_score 做向量相似度）
    # 8.x：可以用 index/similarity（更常见）
    if major >= 8:
        embedding_mapping = {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        }
    else:
        embedding_mapping = {
            "type": "dense_vector",
            "dims": dims,
        }

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
            "embedding": embedding_mapping,
        }
    }

    es.indices.create(index=INDEX, mappings=mappings)


# ----------------------------
# Parsing + embedding
# ----------------------------
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

                parts = [
                    obj.get("capability_id", ""),
                    obj.get("agent_name", ""),
                    obj.get("description", ""),
                    " ".join(keys),
                    " ".join(obj.get("example_queries", []) or []),
                    "required: " + " ".join(obj.get("required_slots", []) or []),
                    "optional: " + " ".join(obj.get("optional_slots", []) or []),
                    "tools: " + " ".join(obj.get("tools_involved", []) or []),
                    obj.get("fallback_prompt", ""),
                    obj.get("relationship_type", ""),
                ]
                obj["search_text"] = " | ".join(
                    [p for p in parts if p and p not in ("required: ", "optional: ", "tools: ")]
                )
                docs.append(obj)

                i = j + 1
                continue
        i += 1
    return docs


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = dashscope.TextEmbedding.call(model=EMBED_MODEL, input=texts)
    if resp.get("status_code") not in (200,):
        raise RuntimeError(f"Embedding failed: {resp}")

    out = resp.get("output", {})
    if "embeddings" in out:
        return [it["embedding"] for it in out["embeddings"]]
    if "data" in out:
        return [it["embedding"] for it in out["data"]]
    raise RuntimeError(f"Unknown embedding response format: {resp}")


def main():
    if not dashscope.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置（.env 中配置）")

    md_text = Path(SEED_MD_ABS).read_text(encoding="utf-8")
    docs = parse_seeds(md_text)
    print("Parsed capability docs:", len(docs))
    if not docs:
        raise RuntimeError("没有从 k-v.md 解析出任何 capability JSON，请检查格式。")

    es = es_client()

    # ✅ 先算一条 embedding 来获取真实维度
    one_emb = embed_texts([docs[0]["search_text"]])[0]
    dims = len(one_emb)
    print("Detected embedding dims:", dims)

    ensure_index(es, dims=dims)

    # ✅ 复用第一条 embedding，剩下的再算
    texts_rest = [d["search_text"] for d in docs[1:]]
    embs = [one_emb] + (embed_texts(texts_rest) if texts_rest else [])

    actions = []
    for d, e in zip(docs, embs):
        d["embedding"] = e
        doc_id = d.get("capability_id") or d.get("agent_name") or str(len(actions))
        actions.append({"_index": INDEX, "_id": doc_id, "_source": d})

    ok, errors = bulk(es, actions, raise_on_error=False)
    es.indices.refresh(index=INDEX)

    print("Indexed:", ok, "errors:", len(errors))
    if errors:
        print("Sample errors (first 3):")
        for e in errors[:3]:
            print(json.dumps(e, ensure_ascii=False)[:1500])

    print("Index:", INDEX)


if __name__ == "__main__":
    main()
