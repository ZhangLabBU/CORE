# core/media.py
import os, re, shutil, hashlib
from typing import List, Dict, Tuple, Optional
from core.utils import to_text

IMG_PAT = re.compile(r"(/[^ \n\t]+?\.(?:jpg|jpeg|png|webp))", re.I)


def extract_image_paths_anywhere(*texts: str) -> List[str]:
    paths = []
    for t in texts:
        s = to_text(t).replace("\\n", "\n")
        paths += IMG_PAT.findall(s)
    out, seen = [], set()
    for p in paths:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def safe_copy_to_keyframe_dir(src_path: str, keyframe_dir: str) -> Optional[str]:
    """
    把任意图片复制到 KEYFRAME_DIR，避免前端 404。
    返回复制后的绝对路径；失败返回 None。
    """
    if not src_path or not os.path.isfile(src_path):
        return None
    os.makedirs(keyframe_dir, exist_ok=True)

    h = hashlib.md5(src_path.encode("utf-8")).hexdigest()[:10]
    base = os.path.basename(src_path)
    dst_name = f"{h}_{base}"
    dst_path = os.path.join(keyframe_dir, dst_name)

    if not os.path.isfile(dst_path):
        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            return None
    return dst_path


def _public_base_url() -> str:
    """
    你必须设置 PUBLIC_BASE_URL，例如：
      http://127.0.0.1:8001
      http://10.200.1.35:8001
    """
    return (os.getenv("PUBLIC_BASE_URL", "") or "").strip().rstrip("/")


def _make_abs(url_path: str) -> str:
    base = _public_base_url()
    if not url_path:
        return url_path
    if url_path.startswith("http://") or url_path.startswith("https://"):
        return url_path
    if not base:
        # 没配置 base，就原样返回（此时仍可能前端跑偏）
        return url_path
    if url_path.startswith("/"):
        return base + url_path
    return base + "/" + url_path


def build_images_payload(
    raw_paths: List[str],
    keyframe_dir: str,
    url_prefix: str = "/files/keyframes/",
    limit: int = 6,
) -> Tuple[List[Dict], str]:
    """
    返回 (images_list, images_from)
    images_list: [{"path": "...", "url": "ABS_URL", "url_rel": "/files/keyframes/.."}]
    """
    out = []
    used = 0
    for p in raw_paths or []:
        if used >= limit:
            break
        copied = safe_copy_to_keyframe_dir(p, keyframe_dir)
        if not copied:
            continue
        rel = url_prefix + os.path.basename(copied)
        out.append(
            {
                "path": copied,
                "url_rel": rel,
                "url": _make_abs(rel),  # ✅ 关键：url 直接给绝对地址
            }
        )
        used += 1
    return out, "copy_to_keyframe_dir"
