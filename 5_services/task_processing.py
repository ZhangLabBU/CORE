# /home/yzt/AgentDNS/src/services/task_processing.py
import os, sys, re, json
from typing import Any, Dict, List,Tuple

# ✅ 确保能 import 到项目根目录的 run_single_agent.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_single_agent import process_agent_request


def _to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return str(x)


def _scratchpad_text(logs: Any) -> str:
    """只拼接 logs 里的 Agent scratchpad（避免混入 Question/Raw Answer）"""
    if not isinstance(logs, list):
        return ""
    parts = []
    for x in logs:
        if isinstance(x, dict):
            sp = x.get("Agent scratchpad")
            if isinstance(sp, str) and sp.strip():
                parts.append(sp)
    return "\n\n".join(parts)


def _looks_like_path(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^(/|\.{1,2}/).*\.(jpg|jpeg|png)$", s, flags=re.I))


def _extract_answer_block_from_logs(logs: Any) -> str:
    """从 logs 里拿到最像最终文本的 Raw Answer/Final Answer"""
    if not isinstance(logs, list):
        return ""
    for entry in reversed(logs):
        if not isinstance(entry, dict):
            continue
        for k in ("Final Answer", "Raw Answer", "final_answer", "raw_answer"):
            v = entry.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _extract_finish_balanced(text: str) -> str:
    """从文本里抓最后一个 Action: Finish(...)，支持括号嵌套/引号"""
    t = _to_text(text).replace("\\n", "\n")
    pos = t.rfind("Action: Finish(")
    if pos == -1:
        return ""
    s = t[pos + len("Action: Finish(") :]

    depth, out = 1, []
    in_quote, esc = None, False

    for ch in s:
        out.append(ch)
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                out.pop()  # pop ')'
                break

    payload = "".join(out).strip()
    if (payload.startswith('"') and payload.endswith('"')) or (
        payload.startswith("'") and payload.endswith("'")
    ):
        payload = payload[1:-1].strip()
    return payload.strip()


def _parse_tool_observations(scratch: str) -> Dict[str, List[str]]:
    """
    解析形如：
      ToolName(...)\nObservation: xxx
    返回：{"ToolName":[xxx], ...}
    """
    t = _to_text(scratch).replace("\\n", "\n")
    pat = re.compile(
        r"(?P<tool>[A-Za-z_]\w*)\([^\)]*\)\s*\nObservation:\s*(?P<obs>.*?)(?=\n[A-Za-z_]\w*\(|\Z)",
        flags=re.DOTALL,
    )
    out: Dict[str, List[str]] = {}
    for m in pat.finditer(t):
        tool = m.group("tool").strip()
        obs = (m.group("obs") or "").strip()
        if not obs:
            continue
        out.setdefault(tool, []).append(obs)
    return out


def _pick_detail_from_scratch(scratch: str) -> Tuple[str, str]:
    """
    返回 (detail, source)
    source 用于 trace 调试
    """
    tools = _parse_tool_observations(scratch)

    def ok(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        if _looks_like_path(s):  # 纯路径直接排除
            return False
        # 你也可以加一个“太短就别要”的阈值，避免只抽到一句话
        # return len(s) >= 20
        return True

    # 1) VideoUnderstanding 最优先（一般给长描述）
    if tools.get("VideoUnderstanding"):
        detail = tools["VideoUnderstanding"][-1].strip()
        if ok(detail):
            return detail, "scratchpad.VideoUnderstanding"

    # 2) KeyframeRecognition 次优先（一般也会给长描述）
    if tools.get("KeyframeRecognition"):
        detail = tools["KeyframeRecognition"][-1].strip()
        if ok(detail):
            return detail, "scratchpad.KeyframeRecognition"

    # 3) 兜底：所有工具的最后一个非路径 observation
    all_obs: List[str] = []
    for tool_name, lst in tools.items():
        for o in lst:
            o = (o or "").strip()
            if ok(o):
                all_obs.append(o)

    if all_obs:
        return all_obs[-1], "scratchpad.last_nonpath_observation"

    return "", ""


def _extract_images_from_scratchpad(scratch: str) -> List[str]:
    scratch = _to_text(scratch).replace("\\n", "\n")
    paths = re.findall(r"(/[^ \n\t]+?\.(?:jpg|jpeg|png))", scratch, flags=re.I)
    uniq, seen = [], set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _normalize_for_frontend(
    agent_name: str, raw_result: Dict[str, Any]
) -> Dict[str, Any]:
    logs_any = raw_result.get("logs", None)
    if isinstance(logs_any, dict):
        logs = [logs_any]           # ✅ 关键：dict 包一层 list
    elif isinstance(logs_any, list):
        logs = logs_any
    else:
        logs = []

    scratch = _scratchpad_text(logs)


    # ✅ answer_block 用于提取 Finish（因为它常常只在 Raw Answer 里）
    answer_block = _extract_answer_block_from_logs(logs)
    if not answer_block:
        answer_block = _to_text(raw_result.get("raw_answer", ""))

    # ✅ 结论：Finish 优先
    conclusion = _extract_finish_balanced(answer_block).strip()

    # ✅ 详细描述：优先从 scratchpad 的工具 Observation 拿（尤其 VideoUnderstanding）
    detail, detail_source = _pick_detail_from_scratch(scratch)

    # ✅ 如果 conclusion 为空：用 raw_answer 兜底（短文本才用，避免把一坨日志当结论）
    if not conclusion:
        ra = _to_text(raw_result.get("raw_answer", "")).strip()
        if ra and len(ra) <= 200:
            conclusion = ra

    img_paths = _extract_images_from_scratchpad(scratch)

    # ✅ 统一展示文本（无“依据/来源”）
    lines: List[str] = []

    if conclusion:
        lines += ["【结论】", conclusion, ""]
    else:
        lines += ["【结论】", "已完成任务，但未提取到明确结论。", ""]

    if detail:
        lines += ["【详细描述】", detail, ""]

    lines += ["【附加内容】", f"- 图片: {len(img_paths)} 张"]

    display_text = "\n".join(lines).strip()

    return {
        "status": "ok",
        "logs": logs,
        "raw_answer": raw_result.get("raw_answer"),
        "display": {"text": display_text},
        "structured": {"images": [{"path": p} for p in img_paths]},
        "trace": {
            "raw_answer": raw_result.get("raw_answer"),
            "logs": logs,
            "raw": raw_result,
            "answer_block_used": answer_block,  # 你现在已经在打印它了
            "detail_source": detail_source,
        },
    }


async def process_task(agent_name: str, prompt: str) -> dict:
    raw_result = await process_agent_request(
        agent_name=agent_name,
        prompt=prompt,
        model_type="openai",
        use_react=False,
    )
    return _normalize_for_frontend(agent_name, raw_result)
