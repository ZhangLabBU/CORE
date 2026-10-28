# src/docstore/media.py
# -*- coding: utf-8 -*-

"""
桥接模块：
让 src.docstore.Report.py 里的 `from .media import ...`
实际复用 /core/media.py 里的实现。
"""

from __future__ import annotations

try:
    # 你已经有的实现
    from core.media import extract_image_paths_anywhere

except Exception as e:
    # 兜底：即使 core.media 暂时不可用，也不让 import 炸
    import re
    from typing import List

    IMG_PAT = re.compile(r"(/[^ \n\t]+?\.(?:jpg|jpeg|png|webp))", re.I)

    def extract_image_paths_anywhere(*texts: str) -> List[str]:
        paths: List[str] = []
        for t in texts:
            s = "" if t is None else str(t)
            s = s.replace("\\n", "\n")
            paths += IMG_PAT.findall(s)

        # 去重保序
        out, seen = [], set()
        for p in paths:
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out


__all__ = ["extract_image_paths_anywhere"]
