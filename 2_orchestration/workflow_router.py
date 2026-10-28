# core/workflow_router.py
import re, os, math, hashlib, random
from core.workflows import WORKFLOWS

_PAT_VIDEO = re.compile(r"(视频|video|mp4|mov|avi|mkv|关键帧|抽帧|截帧|截图|帧)", re.I)
_PAT_KEYFRAME_STRONG = re.compile(r"(关键帧|抽帧|截帧|逐帧|每秒|第\d+秒)", re.I)
_PAT_METEO = re.compile(r"(天气|气温|温度|降雨|下雨|有雨|雨吗|下雪|雪|大风|风力|湿度|预报|气象|预警|雷暴|台风)", re.I)
_PAT_MAP_ENTITY = re.compile(r"(地图|位置|定位|坐标|经纬度|地址|POI|地铁|公交|道路|路口)", re.I)
_PAT_MAP_INTENT = re.compile(r"(导航|路线|怎么去|如何去|到哪|去哪里|附近|周边|离我|距离|多远|路径|最近|规划路线)", re.I)
_PAT_REPORT = re.compile(r"(报告|总结|汇总|生成报告|写报告|形成报告|出一份报告|整理成报告)", re.I)

def _norm(text: str) -> str:
    return (text or "").strip()

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _stable_random_0_1(key: str) -> float:
    if not key:
        return random.random()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return (x % 10_000_000) / 10_000_000.0

def pick_workflow_id(user_input: str, uploaded_video_path: str | None):
    """
    ✅ 返回 dict：
    {
      "selected": wf_id or None,
      "candidates": [{workflow_id, sem_score, net_score, final_score, sem_pct, net_pct, final_pct, reasons}, ...]
    }
    """
    s = _norm(user_input)
    if not s and not uploaded_video_path:
        return {"selected": None, "candidates": []}

    w_sem = float(os.getenv("W_SEM_WF", os.getenv("W_SEM", "0.80")))
    w_net = float(os.getenv("W_NET_WF", os.getenv("W_NET", "0.20")))
    if w_sem < 0: w_sem = 0.0
    if w_net < 0: w_net = 0.0
    if (w_sem + w_net) <= 1e-9:
        w_sem, w_net = 0.8, 0.2

    cands = []

    def add(wf_id: str, sem: float, reasons: list[str], net: float | None = None):
        net = _clamp01(net if net is not None else _stable_random_0_1(f"wf|{wf_id}"))
        sem = _clamp01(sem)
        final = _clamp01(w_sem * sem + w_net * net)
        cands.append({
            "workflow_id": wf_id,
            "sem_score": sem,
            "net_score": net,
            "final_score": final,
            "sem_pct": sem * 100.0,
            "net_pct": net * 100.0,
            "final_pct": final * 100.0,
            "match_pct": final * 100.0,
            "reasons": reasons,
        })

    # -------------------------
    # 1) 有视频：给两个候选（快流 vs 关键帧流），用规则强弱打 sem
    # -------------------------
    if uploaded_video_path:
        has_key_strong = bool(_PAT_KEYFRAME_STRONG.search(s))
        has_video_kw = bool(_PAT_VIDEO.search(s))

        # 强关键帧诉求 → keyframe report sem 更高
        if has_key_strong:
            add("wf_video_keyframe_report", 0.95, ["has_video", "strong_keyframe_intent"])
            add("wf_video_report_fast",     0.75, ["has_video", "fallback_fast"])
        # 普通视频诉求
        elif has_video_kw:
            add("wf_video_report_fast",     0.90, ["has_video", "video_keywords"])
            add("wf_video_keyframe_report", 0.65, ["has_video", "secondary_keyframe"])
        # 上传了但没说啥
        else:
            add("wf_video_report_fast", 0.80, ["has_video", "default_fast"])
            add("wf_video_keyframe_report", 0.55, ["has_video", "secondary_keyframe"])

        cands.sort(key=lambda x: x["final_score"], reverse=True)
        return {"selected": cands[0]["workflow_id"], "candidates": cands}

    # -------------------------
    # 2) 无视频：天气 / 地图 / 联合
    # -------------------------
    has_meteo = bool(_PAT_METEO.search(s))
    has_map_intent = bool(_PAT_MAP_INTENT.search(s))
    has_map_entity = bool(_PAT_MAP_ENTITY.search(s))
    has_report = bool(_PAT_REPORT.search(s))

    if has_meteo and has_map_intent:
        # 联合需求 sem 最高
        add("wf_map_meteo_report", 0.92 + (0.03 if has_report else 0.0), ["meteo", "map_intent", "joint"])
        # 单项作为备选
        add("wf_meteo_report", 0.75, ["meteo", "secondary"])
        add("wf_map_report",   0.70, ["map_intent", "secondary"])

    elif has_meteo:
        add("wf_meteo_report", 0.90 + (0.03 if has_report else 0.0), ["meteo"])
        # 如果同时出现“坐标/地图实体”，也给联合备选
        if has_map_entity:
            add("wf_map_meteo_report", 0.65, ["meteo", "map_entity_secondary"])

    elif has_map_intent:
        add("wf_map_report", 0.88 + (0.03 if has_report else 0.0), ["map_intent"])
        # 有“天气字眼”很弱时也给联合备选（这里没有 meteo，所以不加）

    else:
        return {"selected": None, "candidates": []}

    cands.sort(key=lambda x: x["final_score"], reverse=True)
    return {"selected": cands[0]["workflow_id"], "candidates": cands}

def get_workflow(wf_id: str):
    if wf_id not in WORKFLOWS:
        raise KeyError(f"Unknown workflow: {wf_id}")
    return WORKFLOWS[wf_id]
