# /home/yzt/AgentDNS/configs/ReportAgent/tools.py
import os
import json
import ast
from typing import Any, List, Optional

from src.agents.tools import Tool
from src.docstore.Report import ReportAgent

_agent = ReportAgent()


# -------------------------
# helpers
# -------------------------
def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s


def _resolve_placeholders(x: Any) -> Any:
    """
    只解析字符串里的占位符：
    - $PREV_STEP_SUMMARY
    - $VIDEO_PATH
    - $PREV_STEP_IMAGES
    """
    if not isinstance(x, str):
        return x
    s = _strip_quotes(x)

    if s == "$PREV_STEP_SUMMARY":
        return os.environ.get("PREV_STEP_SUMMARY", "")
    if s == "$VIDEO_PATH":
        return os.environ.get("VIDEO_PATH", "")
    if s == "$PREV_STEP_IMAGES":
        return os.environ.get("PREV_STEP_IMAGES", "[]")
    return x


def _q(s: str) -> str:
    """把字符串转成安全的双引号字符串（用于 stringify_rule）"""
    s = "" if s is None else str(s)
    return json.dumps(s, ensure_ascii=False)


def _fmt_arg(x: Any) -> str:
    """
    stringify_rule 里用：
    - 字符串 -> "xxx"
    - $1 / $2 这种变量 -> 原样输出（不加引号）
    - 其它类型 -> str(x)
    - 若字符串本身是 JSON list，如 ["a","b"]，则保持为 ["a","b"]（不加外层引号）
    """
    if x is None:
        return '""'

    if isinstance(x, str):
        t = x.strip()
        if t.startswith("$"):
            return t

        # ✅ 如果像一个 list literal（JSON/Python），则不要再包一层引号
        if (t.startswith("[") and t.endswith("]")) or (t.startswith("(") and t.endswith(")")):
            return t

        return _q(t)

    return str(x)


def _ensure_list_str(v: Any) -> List[str]:
    """
    将各种可能形态归一成 List[str]：
    - list -> 过滤转 str
    - str:
        - '["/a.jpg","/b.jpg"]' -> json.loads
        - "['/a.jpg']" -> ast.literal_eval
        - "/a.jpg" -> ["/a.jpg"]
        - ""/None -> []
    - 其他 -> []
    """
    if v is None:
        return []

    if isinstance(v, list):
        out = []
        for it in v:
            if it is None:
                continue
            s = str(it).strip()
            if s:
                out.append(s)
        return out

    if isinstance(v, str):
        s = _strip_quotes(v).strip()
        if not s:
            return []

        # 如果是 env 里传来的 JSON list 字符串
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            # 先 json
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pass
            # 再 literal_eval
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pass

        # 普通路径字符串
        return [s]

    return []


def normalize_image_paths(image_paths: Any) -> List[str]:
    """
    先 resolve placeholders，再确保 list[str]
    """
    image_paths = _resolve_placeholders(image_paths)

    # 如果 resolve 后还是 "$2" 这种（极端情况下 planner 直接传），保持原样交给上层
    if isinstance(image_paths, str) and image_paths.strip().startswith("$"):
        # 这种情况 write_report_async 大概率吃不下
        # 我们尽量从 env 拿 PREV_STEP_IMAGES 做兜底
        env_imgs = os.environ.get("PREV_STEP_IMAGES", "[]")
        imgs = _ensure_list_str(env_imgs)
        return imgs

    return _ensure_list_str(image_paths)


# -------------------------
# tool funcs
# -------------------------
async def GenerateReport(logs: str):
    logs = _resolve_placeholders(logs)
    return await _agent.generate_report_async(logs)


async def ReadFile(path_or_text: str):
    path_or_text = _resolve_placeholders(path_or_text)
    return await _agent.read_file_async(path_or_text)


async def GetImagePaths(text: str):
    """
    ✅ 强制返回 List[str]
    """
    text = _resolve_placeholders(text)
    res = await _agent.get_image_paths_async(text)

    # 兜底：有些实现可能返回 str（你已经测到）
    res = _resolve_placeholders(res)
    paths = _ensure_list_str(res)

    # 兼容：如果 agent 内部把结果写进 env，这里也兜底取一次
    if not paths:
        env_imgs = os.environ.get("PREV_STEP_IMAGES", "[]")
        paths = _ensure_list_str(env_imgs)

    return paths


async def WriteReport(report_content: str, image_paths: Optional[Any] = None):
    """
    ✅ image_paths 无论传入 str/list/占位符，都归一成 List[str]
    """
    report_content = _resolve_placeholders(report_content)
    paths = normalize_image_paths(image_paths)
    return await _agent.write_report_async(report_content, paths)


# -------------------------
# tools list
# -------------------------
tools = [
    Tool(
        name="GenerateReport",
        func=GenerateReport,
        description="GenerateReport(logs: str) -> str：根据系统记录/上一步摘要生成【中文】结构化报告。",
        stringify_rule=lambda args: f"GenerateReport({_fmt_arg(args[0])})",
    ),
    Tool(
        name="ReadFile",
        func=ReadFile,
        description="ReadFile(file_path_or_text: str) -> str：读取文件或文本（支持 VIDEO_PATH/占位符）。",
        stringify_rule=lambda args: f"ReadFile({_fmt_arg(args[0])})",
    ),
    Tool(
        name="GetImagePaths",
        func=GetImagePaths,
        description="GetImagePaths(text: str) -> List[str]：从文本中提取图片路径。",
        stringify_rule=lambda args: f"GetImagePaths({_fmt_arg(args[0])})",
    ),
    Tool(
        name="WriteReport",
        func=WriteReport,
        description="WriteReport(report_content: str, image_paths: List[str]=None) -> str：生成 docx 并返回路径。",
        stringify_rule=lambda args: (
            f"WriteReport({_fmt_arg(args[0])}, {_fmt_arg(args[1])})"
            if len(args) > 1 else f"WriteReport({_fmt_arg(args[0])})"
        ),
    ),
]

print("[LOADED TOOLS] configs/ReportAgent/tools.py loaded")
print("[LOADED TOOLS] names =", [t.name for t in tools])
