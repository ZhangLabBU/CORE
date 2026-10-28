# core/normalize.py
import re, ast, json
from typing import Any, Dict, List, Optional, Tuple
from core.utils import (
    to_text,
    truncate,
    looks_like_path,
    looks_like_number,
    looks_like_tool_call,
    contains_error_signal,
    zh_ratio,
)
from .media import extract_image_paths_anywhere

TOOL_OBS_PAT = re.compile(
    r"(?P<tool>[A-Za-z_]\w*)\([^\)]*\)\s*\nObservation:\s*(?P<obs>.*?)(?=\n[A-Za-z_]\w*\(|\Z)",
    flags=re.DOTALL,
)


def parse_tool_observations(scratchpad_text: str) -> Dict[str, List[str]]:
    s = to_text(scratchpad_text).replace("\\n", "\n")
    out: Dict[str, List[str]] = {}
    for m in TOOL_OBS_PAT.finditer(s):
        tool = m.group("tool").strip()
        obs = (m.group("obs") or "").strip()
        if obs:
            out.setdefault(tool, []).append(obs)
    return out


def extract_write_report(text: str) -> str:
    """
    你现在的 ReportAgent scratchpad 里常见是 WriteReport($3, $2)
    这个函数对这种占位符基本没法直接拿到 report 正文，所以不要指望它。
    我们主要靠 tool:GenerateReport 的 Observation 取整篇报告。
    """
    s = to_text(text).replace("\\n", "\n")
    idx = s.rfind("WriteReport(")
    if idx == -1:
        return ""
    tail = s[idx:]
    m = re.search(r"WriteReport\((.*?),\s*(\[[^\]]*\])\s*\)\s*$", tail, flags=re.DOTALL)
    if m:
        return (m.group(1) or "").strip()
    m2 = re.search(r"WriteReport\((.*)\)\s*$", tail, flags=re.DOTALL)
    return (m2.group(1) if m2 else "").strip()


def extract_finish_payload(text: str) -> str:
    """
    取最后一次 Action: Finish(...)
    """
    s = to_text(text).replace("\\n", "\n")
    pos = s.rfind("Action: Finish(")
    if pos == -1:
        return ""
    tail = s[pos + len("Action: Finish(") :]
    depth, out = 1, []
    in_q, esc = None, False
    for ch in tail:
        if esc:
            esc = False
            out.append(ch)
            continue
        if ch == "\\":
            esc = True
            out.append(ch)
            continue
        if in_q:
            out.append(ch)
            if ch == in_q:
                in_q = None
            continue
        if ch in ("'", '"'):
            in_q = ch
            out.append(ch)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    payload = "".join(out).strip()
    if (payload.startswith('"') and payload.endswith('"')) or (
        payload.startswith("'") and payload.endswith("'")
    ):
        payload = payload[1:-1].strip()
    return payload


def pick_keyframe_recognition_result(obs: str) -> str:
    """
    处理你日志里这种：
      Observation: [{'image_path': '...', 'result': '...'}]
    """
    o = (obs or "").strip()
    try:
        parsed = ast.literal_eval(o)
        if isinstance(parsed, list) and parsed:
            it = parsed[0]
            if isinstance(it, dict) and it.get("result"):
                return str(it["result"]).strip()
        if isinstance(parsed, dict) and parsed.get("result"):
            return str(parsed["result"]).strip()
    except Exception:
        pass
    return o


def _looks_like_report(t: str) -> bool:
    s = t or ""
    if "## I." in s or "\n## I." in s:
        return True
    if "I. Overview" in s and "II. Scene" in s:
        return True
    if s.strip().startswith("# Incident Report"):
        return True
    if "Title:" in s and "I. Overview" in s:
        return True
    return False


def collect_text_candidates(
    result_obj: Optional[dict],
    raw_result_str: str,
    logs_obj: Any,
    logs_text: str,
    user_query: str,
) -> List[Dict]:
    """
    return list of candidate dict:
      {"text": "...", "from": "...", "type": "text|error"}
    """
    cands: List[Dict] = []

    # 1) 显式字段
    if isinstance(result_obj, dict):
        for path in [
            ("answer.text", lambda o: (o.get("answer") or {}).get("text")),
            ("text", lambda o: o.get("text")),
            ("output.text", lambda o: (o.get("output") or {}).get("text")),
            (
                "structured.observation",
                lambda o: (o.get("structured") or {}).get("observation"),
            ),
        ]:
            try:
                t = to_text(path[1](result_obj)).strip()
                if t:
                    cands.append({"text": t, "from": path[0], "type": "text"})
            except Exception:
                pass

    # 2) report（WriteReport）— 现在不一定靠谱，但保留
    report = extract_write_report(logs_text)
    if report:
        cands.append({"text": report, "from": "write_report", "type": "text"})

    # 3) tool observations（关键）
    scratch = ""
    if isinstance(logs_obj, dict):
        scratch = to_text(logs_obj.get("Agent scratchpad")).strip()
    elif isinstance(logs_obj, list) and logs_obj and isinstance(logs_obj[0], dict):
        scratch = to_text(logs_obj[0].get("Agent scratchpad")).strip()

    if scratch:
        tools = parse_tool_observations(scratch)

        # KeyframeRecognition -> result
        if "KeyframeRecognition" in tools and tools["KeyframeRecognition"]:
            t = pick_keyframe_recognition_result(tools["KeyframeRecognition"][-1])
            if t:
                cands.append(
                    {"text": t, "from": "tool:KeyframeRecognition", "type": "text"}
                )

        # ✅ ReportAgent 的整篇报告通常在 GenerateReport 的 Observation
        if "GenerateReport" in tools and tools["GenerateReport"]:
            t = tools["GenerateReport"][-1].strip()
            if t:
                cands.append({"text": t, "from": "tool:GenerateReport", "type": "text"})

        # 气象类：Observation 通常比 Finish(...) 更完整
        for tool_name in [
            "AnalyzeWeather",
            "Weather",
            "Meteorology",
            "MeteorologyAgent",
            "MeteorologyTool",
        ]:
            if tool_name in tools and tools[tool_name]:
                t = tools[tool_name][-1].strip()
                if t:
                    cands.append(
                        {"text": t, "from": f"tool:{tool_name}", "type": "text"}
                    )

        # VideoUnderstanding（如果有）
        if "VideoUnderstanding" in tools and tools["VideoUnderstanding"]:
            t = tools["VideoUnderstanding"][-1].strip()
            if t:
                cands.append(
                    {"text": t, "from": "tool:VideoUnderstanding", "type": "text"}
                )

        # 你可以按需扩展其他工具
        for tool_name in ["MapSearch", "Search", "GetNearbyResources"]:
            if tool_name in tools and tools[tool_name]:
                t = tools[tool_name][-1].strip()
                if t:
                    cands.append(
                        {"text": t, "from": f"tool:{tool_name}", "type": "text"}
                    )

    # 4) 从 raw_result_str / logs_text 抓错误（作为 error 候选）
    err_blob = "\n".join([raw_result_str, logs_text])
    if contains_error_signal(err_blob):
        cands.append(
            {"text": truncate(err_blob, 800), "from": "error_detected", "type": "error"}
        )

    # 5) Finish payload（最后兜底）
    fin = extract_finish_payload(logs_text)
    if fin:
        cands.append({"text": fin, "from": "finish_payload", "type": "text"})

    # 6) raw_result_str 兜底（谨慎）
    rr = to_text(raw_result_str).strip()
    if rr:
        cands.append(
            {"text": truncate(rr, 800), "from": "raw_result_str", "type": "text"}
        )

    return cands


def score_text_candidate(text: str, user_query: str, source: str) -> float:
    """
    经验打分：选“像人话 + 信息量足”的
    """
    t = (text or "").strip()
    if not t:
        return -1e9

    # 硬过滤：明显垃圾
    if looks_like_number(t):
        return -1e6
    if looks_like_path(t):
        return -1e6
    if looks_like_tool_call(t) and ("tool:" not in source):
        return -1e5

    # base: length
    L = len(t)
    base = min(L, 2400) / 2400.0  # 允许更长一些（报告）

    # 中文占比
    zhr = zh_ratio(t)
    zh_score = 0.6 * zhr

    # 标点/句子
    punct = 0.2 if any(p in t for p in "。；：，,") else 0.0

    # 关键词相关
    q = (user_query or "").strip()
    kw = [
        "火灾",
        "救援",
        "风速",
        "风向",
        "湿度",
        "降水",
        "翻车",
        "起火",
        "碰撞",
        "行人",
        "护栏",
        "关键帧",
        "视频",
    ]
    rel = 0.0
    for k in kw:
        if k in q and k in t:
            rel += 0.08
    rel = min(rel, 0.5)

    # 惩罚：像日志/JSON
    penalty = 0.0
    if "traceback" in t.lower() or "[ws send]" in t.lower():
        penalty += 0.8
    if t.count("{") + t.count("}") > 10 or t.count('":') > 10:
        penalty += 0.5
    if "/home/" in t and (sum(1 for ch in t if ch == "/") > 6):
        penalty += 0.5

    # source bias：tool observation / write_report 更可信
    bias = 0.0
    if source.startswith("tool:"):
        bias += 0.25
    if source == "write_report":
        bias += 0.20
    if source == "finish_payload":
        bias -= 0.10

    # ✅ 如果看起来是“报告正文”，再额外加分（让整篇报告赢过一句话）
    if _looks_like_report(t):
        bias += 0.55

    # ✅ ReportAgent 最可靠：tool:GenerateReport 的 Observation
    if source == "tool:GenerateReport":
        bias += 0.45

    return base + zh_score + punct + rel + bias - penalty


def select_best_text(cands: List[Dict], user_query: str) -> Tuple[str, str, str]:
    """
    return (best_text, text_from, text_kind)
      text_kind: "text" | "error" | "empty"
    """
    if not cands:
        return "", "none", "empty"

    best = None
    best_score = -1e18

    for c in cands:
        txt = (c.get("text") or "").strip()
        src = c.get("from") or "unknown"
        kind = c.get("type") or "text"

        s = score_text_candidate(txt, user_query, src)
        if kind == "error":
            s -= 0.3

        if s > best_score:
            best_score = s
            best = (txt, src, kind)

    if not best:
        return "", "none", "empty"
    return best[0], best[1], best[2]


def decide_effective_status(best_text: str, text_kind: str, images: List[Dict]) -> str:
    """
    - 有可用 text 或 images -> ok
    - 只有 error -> error
    - 都没有 -> error
    """
    t = (best_text or "").strip()
    if images:
        return "ok"
    if not t:
        return "error"
    if text_kind == "error":
        return "error"
    if t.lower() in {"error", "none", "null", "错误"}:
        return "error"
    return "ok"


def collect_image_candidates(
    result_obj: Optional[dict], logs_text: str, raw_result_str: str
) -> List[str]:
    """
    从结构化字段 + logs + raw_result 扫图
    """
    paths: List[str] = []

    if isinstance(result_obj, dict):
        ans = result_obj.get("answer") or {}
        imgs = ans.get("images")
        if isinstance(imgs, list):
            for it in imgs:
                if isinstance(it, dict) and it.get("path"):
                    paths.append(str(it["path"]))
                elif isinstance(it, str):
                    paths.append(it)

        st = result_obj.get("structured") or {}
        imgs2 = st.get("images")
        if isinstance(imgs2, list):
            for it in imgs2:
                if isinstance(it, dict) and it.get("path"):
                    paths.append(str(it["path"]))
                elif isinstance(it, str):
                    paths.append(it)

    # logs/raw 全局扫
    paths += extract_image_paths_anywhere(logs_text, raw_result_str)

    # 去重保序
    out, seen = [], set()
    for p in paths:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out
