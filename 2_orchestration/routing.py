# core/routing.py
import os, math, random, hashlib
from typing import Any, Dict, List, Tuple, Optional

from elasticsearch import Elasticsearch
from Pre_processing.langchain_es_bm25_demo import CapabilityHybridRetriever


# -----------------------
# small utils
# -----------------------
def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _stable_random_0_1(key: str) -> float:
    if not key:
        return random.random()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return (x % 10_000_000) / 10_000_000.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clamp100(x: float) -> float:
    return max(0.0, min(100.0, float(x)))


def _softmax_temp(xs: List[float], T: float = 0.05) -> List[float]:
    if not xs:
        return []
    T = max(1e-6, float(T))
    m = max(xs)
    exps = [math.exp((x - m) / T) for x in xs]
    s = sum(exps)
    return [e / (s + 1e-12) for e in exps]


def _make_key(name: str, cap: str) -> str:
    return f"{name}|{cap or ''}"


# -----------------------
# net score (0~1 越大越好)
# -----------------------
def compute_net_score(src: dict, *, fallback_key: str = "") -> Tuple[float, bool]:
    """
    返回 (net_score, net_valid)
    net_valid=False 表示用的稳定随机兜底
    """
    net = _safe_float(src.get("net_score"))
    if net is not None:
        return _clamp01(net), True

    lat = _safe_float(src.get("lat_score"))
    q = _safe_float(src.get("queue_score"))
    if lat is not None or q is not None:
        lat = lat if lat is not None else 0.0
        q = q if q is not None else 0.0
        net = 0.7 * lat + 0.3 * q
        return _clamp01(net), True

    net = _stable_random_0_1(fallback_key)
    return _clamp01(net), False


# -----------------------
# score normalization (关键：sem 不一定是 0~1)
# -----------------------
def normalize_sem_scores(sem_by_key: Dict[str, float]) -> Dict[str, float]:
    """
    把任意尺度 sem 分数压到 0~1
    - 如果 sem 本身就是 0~1：几乎不变
    - 如果 sem 是 BM25/混合分（可能 > 1）：做 min-max + sigmoid-like 稳定压缩
    """
    if not sem_by_key:
        return {}

    vals = [float(v) for v in sem_by_key.values()]
    vmin = min(vals)
    vmax = max(vals)

    # 全一样（比如全 0）
    if abs(vmax - vmin) < 1e-9:
        return {k: (1.0 if vmax > 0 else 0.0) for k in sem_by_key}

    # min-max -> 0..1
    out = {}
    for k, v in sem_by_key.items():
        x = (float(v) - vmin) / (vmax - vmin)
        out[k] = _clamp01(x)
    return out


# -----------------------
# retrieval / scoring
# -----------------------
async def rag_retrieval(
    query: str,
    *,
    es_client: Elasticsearch,
    index_name: str,
    all_limit: int = 50,
    route_topk: int = 10,
    w_sem: float = 0.80,
    w_net: float = 0.20,
) -> List[Dict]:
    """
    返回 candidates（全量 agent 列表，但带上 sem/net/final 分数并排序）
    """
    # 1) 拉全量 agents
    try:
        resp_all = es_client.search(
            index=index_name,
            body={"query": {"match_all": {}}, "size": int(all_limit)},
        )
    except Exception:
        resp_all = es_client.search(
            index=index_name,
            query={"match_all": {}},
            size=int(all_limit),
        )

    hits = (resp_all.get("hits") or {}).get("hits") or []
    if not hits:
        return []

    all_agents: List[Dict] = []
    for h in hits:
        src = h.get("_source") or {}
        agent_name = src.get("agent_name") or src.get("name") or src.get("agent") or src.get("id")
        capability = src.get("capability_id") or src.get("capability") or src.get("capabilityId")
        desc = src.get("description") or src.get("page_content") or src.get("text") or ""

        # 过滤掉 name 为空的垃圾记录（不然 main 执行会炸）
        if not agent_name:
            continue

        fallback_key = f"{agent_name}|{capability}"
        net, net_valid = compute_net_score(src, fallback_key=fallback_key)

        all_agents.append({
            "agent_name": agent_name,
            "capability": capability,
            "description": desc,
            "net_score": net,               # 0~1
            "net_pct": net * 100.0,         # 0~100
            "net_valid": net_valid,
        })

    if not all_agents:
        return []

    # 2) 对 query 算语义/混合分（命中部分）
    route_topk = max(1, min(int(route_topk), len(all_agents)))
    retriever = CapabilityHybridRetriever(es=es_client, index=index_name, k=route_topk)
    docs = retriever.invoke(query) or []

    # 两级 map：name|cap 和 name-only
    sem_map_full: Dict[str, float] = {}
    sem_map_name: Dict[str, float] = {}

    for d in docs:
        md = getattr(d, "metadata", None) or {}

        name = md.get("agent_name") or md.get("name") or md.get("agent") or md.get("id")
        cap = (
            md.get("capability_id")
            or md.get("capability")
            or md.get("capabilityId")
            or md.get("capability_name")
            or ""
        )
        if not name:
            continue

        sem = (
            md.get("agent_sem_score")
            or md.get("sem_score")
            or md.get("semantic_score")
            or md.get("fused_score")
            or md.get("hybrid_score")
            or md.get("score")
            or md.get("raw_score")
        )
        sem = float(sem or 0.0)

        k_full = _make_key(name, cap)
        sem_map_full[k_full] = max(sem_map_full.get(k_full, 0.0), sem)
        sem_map_name[name] = max(sem_map_name.get(name, 0.0), sem)

    # ✅ 归一化：避免 sem 分数尺度不一致导致融合失真
    sem_norm_full = normalize_sem_scores(sem_map_full)
    sem_norm_name = normalize_sem_scores(sem_map_name)

    # 3) 合并：全量 agent + sem_norm + net -> final
    w_sem = float(w_sem)
    w_net = float(w_net)
    if w_sem < 0: w_sem = 0.0
    if w_net < 0: w_net = 0.0
    if (w_sem + w_net) <= 1e-9:
        w_sem, w_net = 0.8, 0.2

    results: List[Dict] = []

    for a in all_agents:
        name = a["agent_name"]
        cap = a.get("capability") or ""
        k_full = _make_key(name, cap)

        # full key 优先，其次 name-only
        sem_raw = float(sem_map_full.get(k_full, sem_map_name.get(name, 0.0)) or 0.0)
        sem_norm = float(sem_norm_full.get(k_full, sem_norm_name.get(name, 0.0)) or 0.0)

        net = float(a.get("net_score") or 0.0)
        net = _clamp01(net)

        final = (w_sem * sem_norm + w_net * net)
        final = _clamp01(final)

        cand = {
            "agent_name": name,
            "capability": a.get("capability"),
            "description": a.get("description", ""),

            # raw 分数（可用于 debug）
            "sem_raw": sem_raw,
            "sem_score": sem_norm,          # ✅ sem_score 统一定义为 0~1（归一化后）
            "net_score": net,
            "final_score": final,

            "sem_pct": _clamp100(sem_norm * 100.0),
            "net_pct": _clamp100(net * 100.0),
            "final_pct": _clamp100(final * 100.0),

            # 兼容旧字段
            "match_pct": _clamp100(final * 100.0),

            # ✅ 给 main/前端一个统一的 confidence 包
            "confidence": {
                "sem_score": sem_norm,
                "net_score": net,
                "final_score": final,
                "sem_pct": _clamp100(sem_norm * 100.0),
                "net_pct": _clamp100(net * 100.0),
                "final_pct": _clamp100(final * 100.0),
                "match_pct": _clamp100(final * 100.0),
            },
        }
        results.append(cand)

    results.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    return results


# -----------------------
# gate: decide fallback or run agent
# -----------------------
def gate_candidates(candidates: List[Dict]) -> Dict[str, Any]:
    """
    输入：按 final_score 排序的 candidates
    输出：gate 决策信息（main 用它决定是否 fallback）
    """
    if not candidates:
        return {"should_fallback": True, "reason": "no_candidates", "selected": None, "debug": {}}

    softmax_t = float(os.getenv("SOFTMAX_T", "0.05"))
    min_final = float(os.getenv("MIN_FINAL_SCORE", "0.10"))
    min_p_gap = float(os.getenv("MIN_P_GAP", "0.05"))
    min_p_ratio = float(os.getenv("MIN_P_RATIO", "1.15"))
    k_gate = int(os.getenv("K_GATE", "5"))
    eps = 1e-9

    top = candidates[: max(1, min(k_gate, len(candidates)))]
    top_scores = [float(c.get("final_score") or 0.0) for c in top]

    # 全 0 或全一样时 softmax 会很“平均”，这里保底
    if max(top_scores) - min(top_scores) < 1e-9:
        probs = [1.0] + [0.0] * (len(top_scores) - 1)
    else:
        probs = _softmax_temp(top_scores, T=softmax_t)

    p1 = probs[0] if len(probs) > 0 else 1.0
    p2 = probs[1] if len(probs) > 1 else 0.0

    best = candidates[0]
    best_final = float(best.get("final_score") or 0.0)
    p_gap = p1 - p2
    p_ratio = p1 / (p2 + eps) if p2 > 0 else float("inf")

    low_abs = best_final < min_final
    low_sep = (len(candidates) > 1 and p_gap < min_p_gap and p_ratio < min_p_ratio)

    if low_abs:
        return {
            "should_fallback": True,
            "reason": "low_confidence_abs",
            "selected": best,
            "debug": {
                "softmax_t": softmax_t,
                "min_final": min_final,
                "min_p_gap": min_p_gap,
                "min_p_ratio": min_p_ratio,
                "k_gate": k_gate,
                "best_final": best_final,
                "p1": p1,
                "p2": p2,
                "p_gap": p_gap,
                "p_ratio": p_ratio,
            },
        }

    return {
        "should_fallback": False,
        "reason": "tie_break_use_top1" if low_sep else "",
        "selected": best,
        "debug": {
            "softmax_t": softmax_t,
            "min_final": min_final,
            "min_p_gap": min_p_gap,
            "min_p_ratio": min_p_ratio,
            "k_gate": k_gate,
            "best_final": best_final,
            "p1": p1,
            "p2": p2,
            "p_gap": p_gap,
            "p_ratio": p_ratio,
        },
    }


# -----------------------
# ONE ENTRY: route()
# -----------------------
async def route(
    user_query: str,
    top_k: int,
    *,
    es_client: Elasticsearch,
    index_name: str,
    return_all: bool = False,  # ✅ 是否返回全量候选给 debug
) -> Dict[str, Any]:
    """
    main.py 调这个就行
    """
    candidates_all = await rag_retrieval(
        user_query,
        es_client=es_client,
        index_name=index_name,
        all_limit=int(os.getenv("ALL_AGENTS_LIMIT", "50")),
        route_topk=int(os.getenv("ROUTE_TOPK", "10")),
        w_sem=float(os.getenv("W_SEM", "0.80")),
        w_net=float(os.getenv("W_NET", "0.20")),
    )

    gate = gate_candidates(candidates_all)
    selected = gate.get("selected")

    # ✅ top_k 生效：默认只返回前 top_k 给前端展示
    top_k = int(top_k or 5)
    top_k = max(1, min(top_k, len(candidates_all))) if candidates_all else 0
    candidates = candidates_all[:top_k] if candidates_all else []

    return {
        "candidates": candidates_all if return_all else candidates,
        "candidates_all_n": len(candidates_all),
        "selected": selected,
        "should_fallback": gate.get("should_fallback", False),
        "fallback_reason": gate.get("reason", ""),
        "gate_debug": gate.get("debug", {}),
        "weights": {
            "w_sem": float(os.getenv("W_SEM", "0.80")),
            "w_net": float(os.getenv("W_NET", "0.20")),
        },
        "limits": {
            "top_k": top_k,
            "route_topk": int(os.getenv("ROUTE_TOPK", "10")),
            "all_limit": int(os.getenv("ALL_AGENTS_LIMIT", "50")),
        },
    }
