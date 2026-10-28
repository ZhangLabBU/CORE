# core/utils.py
import json
import re
from typing import Any

def to_text(x: Any) -> str:
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

def truncate(s: Any, n: int = 800) -> str:
    t = to_text(s)
    return t if len(t) <= n else t[:n] + f"...[truncated {len(t)-n} chars]"

_PATH_PAT = re.compile(r"(^/|[A-Za-z]:\\).+\.(jpg|jpeg|png|webp|mp4|mov|avi|mkv|json|jsonl|txt)$", re.I)

def looks_like_path(t: str) -> bool:
    t = (t or "").strip()
    if not t:
        return False
    if "/home/" in t or t.startswith("/") or re.match(r"^[A-Za-z]:\\", t):
        return True
    return bool(_PATH_PAT.search(t))

def looks_like_number(t: str) -> bool:
    t = (t or "").strip()
    if not t:
        return False
    # 纯数字/浮点/百分号/短数值
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?%?", t)) or (len(t) <= 3 and t.isdigit())

def looks_like_tool_call(t: str) -> bool:
    t = (t or "").strip()
    if not t:
        return False
    # ToolName(...)
    return bool(re.match(r"^[A-Za-z_]\w*\(.*\)$", t))

def contains_error_signal(blob: str) -> bool:
    s = (blob or "").lower()
    bad = [
        "traceback", "exception", "error code:", "access denied", "insufficient_quota",
        "output_parsing_failure", "connection error", "not found", "keyerror", "timeout",
        "free allocated quota exceeded",
    ]
    return any(k in s for k in bad)

def zh_ratio(s: str) -> float:
    s = s or ""
    if not s:
        return 0.0
    zh = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return zh / max(1, len(s))
