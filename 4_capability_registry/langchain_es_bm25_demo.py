# test.py
# 在原有“能力检索 + Agent 聚合 + Agent 阈值过滤”的基础上：
# ✅ 加入：假数据生成（Agent级时延/排队）+ 每5秒更新 + 网络分 + agent_final_score 排序 + 输出展示

import os
import json
import time
import random
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from pydantic import Field, ConfigDict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import dashscope


# ============ 环境变量（沿用你原始代码）============
ENV_PATH = "/home/yzt/AgentDNS/src/react/.env"
load_dotenv(dotenv_path=ENV_PATH)

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD")

print(f"DASHSCOPE_API_KEY: {dashscope.api_key}")
print(f"ES_URL: {ES_URL}")

CAP_INDEX = os.getenv("CAP_INDEX", "capability_registry")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v2")
KNN_THRESHOLD = float(os.getenv("KNN_THRESHOLD", "0.7"))

# Agent 语义阈值：先过滤掉明显不相关的 Agent
AGENT_THRESHOLD = float(os.getenv("AGENT_THRESHOLD", "0.5"))

# QoS 权重（可在 .env 里覆盖）
W_SEM = float(os.getenv("W_SEM", "0.70"))
W_LAT = float(os.getenv("W_LAT", "0.20"))
W_Q = float(os.getenv("W_Q", "0.10"))

# 假数据更新间隔（秒）
STATUS_UPDATE_INTERVAL_S = float(os.getenv("STATUS_UPDATE_INTERVAL_S", "5"))

# 假数据持久化文件（为了多次运行脚本也能看到“每5秒更新”的效果）
STATUS_STORE_PATH = os.getenv("STATUS_STORE_PATH", "/tmp/agent_status_fake.json")


def embed_query(text: str) -> list[float]:
    """把 query 文本转成向量（DashScope embedding）"""
    resp = dashscope.TextEmbedding.call(model=EMBED_MODEL, input=[text])
    if resp.get("status_code") != 200:
        raise RuntimeError(f"Embedding failed: {resp}")

    out = resp.get("output", {})
    if "embeddings" in out:
        return out["embeddings"][0]["embedding"]
    if "data" in out:
        return out["data"][0]["embedding"]
    raise RuntimeError(f"Unknown embedding response format: {resp}")


class CapabilityHybridRetriever(BaseRetriever):
    """
    Hybrid = BM25 + KNN(embedding) 融合 + KNN_THRESHOLD 门控
    Step1: capability 级 fused_score
    Step2: 按 agent_name 聚合（max）得到 agent_sem_score
    Step3: agent_sem_score >= AGENT_THRESHOLD 过滤
    Step4: 加入 Agent 级 QoS 假数据（latency_ms, queue_len），映射为网络分
    Step5: agent_final_score = W_SEM*agent_sem + W_LAT*lat + W_Q*queue，并排序输出
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    es: Elasticsearch = Field(exclude=True)
    index: str

    # Top-k Agents
    k: int = 5
    bm25_n: int = 20
    knn_n: int = 20
    alpha: float = 0.6  # capability fused: (1-alpha)*bm25_norm + alpha*knn_norm

    cap_per_agent_keep: int = 3
    agent_threshold: float = AGENT_THRESHOLD

    # QoS
    w_sem: float = W_SEM
    w_lat: float = W_LAT
    w_q: float = W_Q
    status_update_interval_s: float = STATUS_UPDATE_INTERVAL_S
    status_store_path: str = STATUS_STORE_PATH

    # ----------------- QoS 假数据（Agent级）-----------------
    def _load_status(self) -> dict:
        p = Path(self.status_store_path)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}

    def _save_status(self, data: dict) -> None:
        p = Path(self.status_store_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # 写失败不影响主流程
            pass

    def _refresh_agent_status(self, agent_names: set[str]) -> dict:
        """
        为 agent_names 维护假数据：
        - latency_ms: 0~100（float）
        - queue_len: 0~5（int）
        每 5 秒（status_update_interval_s）更新一次（随机游走），并持久化到文件
        """
        now = time.time()
        store = self._load_status()

        # store 结构：
        # {
        #   "KeyframeAgent": {"latency_ms": 23.4, "queue_len": 2, "ts": 173xxxxxxx},
        #   ...
        # }
        # 只要“任意一个 agent 的 ts”已过期，我们就整体刷新一次（简单策略）
        # 也可以按 agent 分别判断，这里取简单/一致的更新节奏。
        need_update = True
        any_ts = None
        for a, st in store.items():
            if isinstance(st, dict) and "ts" in st:
                any_ts = float(st.get("ts", 0.0))
                break
        if any_ts is not None and (now - any_ts) < self.status_update_interval_s:
            need_update = False

        if need_update:
            # 为了“同一个Agent变化平滑”，使用随机游走
            for a in agent_names:
                st = store.get(a)
                if not isinstance(st, dict):
                    # 初始化
                    store[a] = {
                        "latency_ms": random.uniform(0.0, 100.0),
                        "queue_len": random.randint(0, 5),
                        "ts": now,
                    }
                else:
                    lat = float(st.get("latency_ms", random.uniform(0.0, 100.0)))
                    q = int(st.get("queue_len", random.randint(0, 5)))

                    # 随机游走并裁剪到范围
                    lat = max(0.0, min(100.0, lat + random.uniform(-10.0, 10.0)))
                    q = max(0, min(5, q + random.choice([-1, 0, 1])))

                    st["latency_ms"] = lat
                    st["queue_len"] = q
                    st["ts"] = now
                    store[a] = st

            # 也清理掉不再出现的 agent（可选：这里保留最近见过的也行）
            # 为了让文件不无限增长：只保留本次看到的 agent + 既有的也可以
            # 这里做温和策略：不清理（你可按需打开）
            self._save_status(store)

        # 确保 agent_names 都有值
        for a in agent_names:
            if a not in store:
                store[a] = {
                    "latency_ms": random.uniform(0.0, 100.0),
                    "queue_len": random.randint(0, 5),
                    "ts": now,
                }
        return store

    @staticmethod
    def _lat_score(latency_ms: float) -> float:
        # latency: 0~100 -> score: 1~0
        return max(0.0, min(1.0, 1.0 - (float(latency_ms) / 100.0)))

    @staticmethod
    def _queue_score(queue_len: int) -> float:
        # queue: 0~5 -> score: 1~0
        return max(0.0, min(1.0, 1.0 - (int(queue_len) / 5.0)))

    # ----------------- Retriever 主流程 -----------------
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        # 1) BM25 候选
        bm25_resp = self.es.search(
            index=self.index,
            size=self.bm25_n,
            query={
                "bool": {
                    "should": [
                        {"match": {"search_text": query}},
                        {"match": {"description": query}},
                        {"match": {"keys": query}},
                    ]
                }
            },
        )
        bm25_hits = bm25_resp["hits"]["hits"]
        bm25_score = {h["_id"]: float(h["_score"]) for h in bm25_hits}

        # 2) KNN 候选（script_score + cosineSimilarity）
        qvec = embed_query(query)
        knn_resp = self.es.search(
            index=self.index,
            size=self.knn_n,
            query={
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": qvec},
                    },
                }
            },
        )
        knn_hits = knn_resp["hits"]["hits"]
        knn_score = {h["_id"]: float(h["_score"]) for h in knn_hits}

        # 2.5) 语义门控：KNN top1 必须 >= KNN_THRESHOLD
        if knn_hits:
            best_knn_id = knn_hits[0]["_id"]
            best_knn_raw = float(knn_hits[0]["_score"])
        else:
            best_knn_id = None
            best_knn_raw = 0.0

        if best_knn_raw < KNN_THRESHOLD:
            return [
                Document(
                    page_content="NO_ROUTE",
                    metadata={
                        "reason": "KNN top1 below threshold",
                        "best_knn_id": best_knn_id,
                        "best_knn_raw": best_knn_raw,
                        "knn_threshold": KNN_THRESHOLD,
                    },
                )
            ]

        # 3) capability 融合：min-max 归一化后加权
        def minmax_norm(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            vals = list(scores.values())
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return {k: 1.0 for k in scores}
            return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

        bm25_norm = minmax_norm(bm25_score)
        knn_norm = minmax_norm(knn_score)
        all_ids = set(bm25_norm) | set(knn_norm)

        # src_map：取 agent_name/capability_id/url/slots 等
        src_map = {}
        for h in bm25_hits:
            src_map[h["_id"]] = h["_source"]
        for h in knn_hits:
            src_map.setdefault(h["_id"], h["_source"])

        fused_caps = []
        for _id in all_ids:
            s_b = bm25_norm.get(_id, 0.0)
            s_k = knn_norm.get(_id, 0.0)
            s_f = (1 - self.alpha) * s_b + self.alpha * s_k
            fused_caps.append((_id, s_f, s_b, s_k))

        fused_caps.sort(key=lambda x: x[1], reverse=True)

        # per-capability KNN_THRESHOLD 过滤：避免纯关键词顶上来
        fused_caps = [t for t in fused_caps if knn_score.get(t[0], 0.0) >= KNN_THRESHOLD]

        if not fused_caps:
            return [
                Document(
                    page_content="NO_ROUTE",
                    metadata={
                        "reason": "No candidates after per-capability KNN_THRESHOLD filtering",
                        "best_knn_raw": best_knn_raw,
                        "knn_threshold": KNN_THRESHOLD,
                    },
                )
            ]

        # 展开 capability 条目
        cap_entries = []
        for _id, s_f, s_b, s_k in fused_caps:
            src = src_map.get(_id, {})
            cap_entries.append(
                {
                    "_id": _id,
                    "capability_id": src.get("capability_id"),
                    "agent_name": src.get("agent_name") or "UNKNOWN_AGENT",
                    "url": src.get("url"),
                    "required_slots": src.get("required_slots"),
                    "optional_slots": src.get("optional_slots"),
                    "tools_involved": src.get("tools_involved"),
                    "page_content": src.get("search_text", "") or src.get("description", ""),
                    "fused_score": float(s_f),
                    "bm25_norm": float(s_b),
                    "knn_norm": float(s_k),
                    "bm25_raw": float(bm25_score.get(_id, 0.0)),
                    "knn_raw": float(knn_score.get(_id, 0.0)),
                }
            )

        # 4) Agent 聚合（max）
        by_agent: dict[str, list[dict]] = {}
        for e in cap_entries:
            by_agent.setdefault(e["agent_name"], []).append(e)

        agent_entries = []
        for agent, entries in by_agent.items():
            entries.sort(key=lambda x: x["fused_score"], reverse=True)
            best = entries[0]
            agent_entries.append(
                {
                    "agent_name": agent,
                    "agent_url": best.get("url"),
                    "agent_sem_score": float(best["fused_score"]),  # max 聚合结果
                    "best_capability_id": best.get("capability_id"),
                    "best_required_slots": best.get("required_slots"),
                    "best_optional_slots": best.get("optional_slots"),
                    "best_tools_involved": best.get("tools_involved"),
                    "best_page_content": best.get("page_content", ""),
                    "best_scores": {
                        "fused_score": best["fused_score"],
                        "bm25_norm": best["bm25_norm"],
                        "knn_norm": best["knn_norm"],
                        "bm25_raw": best["bm25_raw"],
                        "knn_raw": best["knn_raw"],
                    },
                    "top_capabilities": [
                        {
                            "capability_id": x.get("capability_id"),
                            "fused_score": x["fused_score"],
                            "bm25_norm": x["bm25_norm"],
                            "knn_norm": x["knn_norm"],
                            "bm25_raw": x["bm25_raw"],
                            "knn_raw": x["knn_raw"],
                            "required_slots": x.get("required_slots"),
                            "url": x.get("url"),
                        }
                        for x in entries[: self.cap_per_agent_keep]
                    ],
                }
            )

        # 5) Agent 语义阈值过滤（agent_sem_score >= AGENT_THRESHOLD）
        before = len(agent_entries)
        agent_entries = [a for a in agent_entries if a["agent_sem_score"] >= self.agent_threshold]
        if not agent_entries:
            return [
                Document(
                    page_content="NO_ROUTE",
                    metadata={
                        "reason": "No agents after AGENT_THRESHOLD filtering",
                        "agent_threshold": self.agent_threshold,
                        "agents_before_filter": before,
                        "best_knn_raw": best_knn_raw,
                        "knn_threshold": KNN_THRESHOLD,
                    },
                )
            ]

        # 6) 加入 QoS 假数据 + 网络分 + agent_final_score
        agent_names = {a["agent_name"] for a in agent_entries}
        status_store = self._refresh_agent_status(agent_names)

        for a in agent_entries:
            st = status_store.get(a["agent_name"], {"latency_ms": 100.0, "queue_len": 5, "ts": time.time()})
            latency_ms = float(st.get("latency_ms", 100.0))
            queue_len = int(st.get("queue_len", 5))

            lat_score = self._lat_score(latency_ms)
            q_score = self._queue_score(queue_len)

            # 最终分：语义 + 网络
            final = (self.w_sem * a["agent_sem_score"]) + (self.w_lat * lat_score) + (self.w_q * q_score)

            a["latency_ms"] = latency_ms
            a["queue_len"] = queue_len
            a["lat_score"] = lat_score
            a["queue_score"] = q_score
            a["agent_final_score"] = float(final)
            a["status_ts"] = float(st.get("ts", 0.0))

        # 7) 按 agent_final_score 排序，取 Top-k
        agent_entries.sort(key=lambda x: x["agent_final_score"], reverse=True)
        agent_entries = agent_entries[: self.k]

        # 8) 输出：Agent级 Document，metadata 带全量可解释信息
        docs = []
        for a in agent_entries:
            docs.append(
                Document(
                    page_content=a["best_page_content"],
                    metadata={
                        "agent_name": a["agent_name"],
                        "agent_url": a["agent_url"],

                        # 语义聚合分
                        "agent_sem_score": a["agent_sem_score"],

                        # QoS
                        "latency_ms": a["latency_ms"],
                        "queue_len": a["queue_len"],
                        "lat_score": a["lat_score"],
                        "queue_score": a["queue_score"],
                        "status_ts": a["status_ts"],

                        # 最终分（排序依据）
                        "agent_final_score": a["agent_final_score"],

                        # 参数/能力解释
                        "best_capability_id": a["best_capability_id"],
                        "best_required_slots": a["best_required_slots"],
                        "best_optional_slots": a["best_optional_slots"],
                        "best_tools_involved": a["best_tools_involved"],
                        "best_scores": a["best_scores"],
                        "top_capabilities": a["top_capabilities"],

                        # 阈值/权重
                        "knn_threshold": KNN_THRESHOLD,
                        "agent_threshold": self.agent_threshold,
                        "weights": {"W_SEM": self.w_sem, "W_LAT": self.w_lat, "W_Q": self.w_q},
                        "status_update_interval_s": self.status_update_interval_s,
                        "status_store_path": self.status_store_path,
                    },
                )
            )

        return docs


def main():
    print("KNN_THRESHOLD =", KNN_THRESHOLD)
    print("AGENT_THRESHOLD =", AGENT_THRESHOLD)
    print(f"WEIGHTS: W_SEM={W_SEM}, W_LAT={W_LAT}, W_Q={W_Q}")
    print(f"STATUS_UPDATE_INTERVAL_S = {STATUS_UPDATE_INTERVAL_S}")
    print(f"STATUS_STORE_PATH = {STATUS_STORE_PATH}")

    if not ES_PASSWORD:
        raise RuntimeError("ES_PASSWORD is missing in .env")
    if not dashscope.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing in .env")
    if not ES_URL:
        raise RuntimeError("ES_URL is missing in .env")

    es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD))

    retriever = CapabilityHybridRetriever(
        es=es,
        index=CAP_INDEX,
        k=5,
        bm25_n=20,
        knn_n=20,
        alpha=0.6,
        agent_threshold=AGENT_THRESHOLD,
        w_sem=W_SEM,
        w_lat=W_LAT,
        w_q=W_Q,
        status_update_interval_s=STATUS_UPDATE_INTERVAL_S,
        status_store_path=STATUS_STORE_PATH,
    )

    query = "分析所有关键帧"
    results = retriever.invoke(query)

    print(f"\nQuery: {query}")

    if results and results[0].page_content == "NO_ROUTE":
        m = results[0].metadata
        print("NO_ROUTE:", m.get("reason"))
        # 尽量打印关键诊断信息
        if "best_knn_raw" in m:
            print(f"  best_knn_raw={m.get('best_knn_raw')} knn_threshold={m.get('knn_threshold', KNN_THRESHOLD)}")
        if "agent_threshold" in m:
            print(f"  agent_threshold={m.get('agent_threshold')} agents_before_filter={m.get('agents_before_filter')}")
        return

    print(f"Top {len(results)} results (AGENT-AGGREGATED + QoS, sorted by agent_final_score):\n")
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(
            f"[{i}] agent={m['agent_name']}  final={m['agent_final_score']:.4f}  "
            f"(sem={m['agent_sem_score']:.4f}, lat_s={m['lat_score']:.4f}, q_s={m['queue_score']:.4f})  "
            f"latency_ms={m['latency_ms']:.1f} queue={m['queue_len']}  url={m.get('agent_url')}"
        )
        print(f"    best_capability={m.get('best_capability_id')}  required={m.get('best_required_slots')}")
        bs = m.get("best_scores", {})
        print(
            "    best_scores:"
            f" fused={bs.get('fused_score', 0.0):.4f}"
            f" bm25_norm={bs.get('bm25_norm', 0.0):.4f}"
            f" knn_norm={bs.get('knn_norm', 0.0):.4f}"
            f" bm25_raw={bs.get('bm25_raw', 0.0):.4f}"
            f" knn_raw={bs.get('knn_raw', 0.0):.4f}"
        )

        top_caps = m.get("top_capabilities", [])
        if top_caps:
            print("    top_capabilities (within this agent):")
            for c in top_caps:
                print(
                    f"      - {c.get('capability_id')}  fused={c.get('fused_score',0.0):.4f} "
                    f"(bm25_norm={c.get('bm25_norm',0.0):.4f}, knn_norm={c.get('knn_norm',0.0):.4f}) "
                    f"bm25_raw={c.get('bm25_raw',0.0):.4f} knn_raw={c.get('knn_raw',0.0):.4f}"
                )

        # 状态更新时间戳（便于你确认“5秒更新”是否生效）
        ts = float(m.get("status_ts", 0.0))
        if ts > 0:
            print(f"    status_ts={ts:.3f}  (age={time.time() - ts:.1f}s)")
        print()


if __name__ == "__main__":
    main()
