#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import time
import json
import traceback
import ast
import hashlib
import shutil
import re

from pathlib import Path
from typing import Any, Optional, Dict
from uvicorn.protocols.utils import ClientDisconnected

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from elasticsearch import Elasticsearch

from core.utils import to_text, truncate
from core.normalize import (
    collect_text_candidates,
    select_best_text,
    decide_effective_status,
    collect_image_candidates,
)
from core.media import build_images_payload

# 你已有的函数（不动）
from services.query_rewrite import rewrite_query_with_llm
from services.task_processing import process_task
from core.workflow_router import pick_workflow_id, get_workflow
from openclaw_client import ping_openclaw
# ✅ routing：推荐用 route()，若你暂时只有 rag_retrieval 也兼容
try:
    from core.routing import route  # 推荐：route() -> {candidates, selected, should_fallback...}
    HAS_ROUTE_FUNC = True
except Exception:
    HAS_ROUTE_FUNC = False
    from core.routing import rag_retrieval  # 兼容：只返回 candidates

# =========================
# UI-only fake routing (DO NOT AFFECT WORKFLOW PICK)
# =========================

# ✅ 这里填你项目里“已有的那几个 agent”，至少 5 个
# 建议用实际存在且 process_task 能识别的名字
UI_AGENT_POOL = [
    "ReportAgent",
    "VideoAgent",
    "KeyframeAgent",
    "MeteorologyAgent",
    "MapAgent",
    # 你还有的话继续加：比如 "SearchAgent" "TrafficAgent" 等
]

def _stable01(key: str) -> float:
    import hashlib
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 10000) / 10000.0

def build_fake_step_candidates(
    *,
    request_id: str,
    step_id: str,
    exec_agent: str,
    pool: list[str],
    k: int = 5,
) -> list[dict]:
    """
    ✅ 只用于前端展示的假候选，不影响任何路由/工作流选择。
    规则：
      1) 返回 k=5 个候选，全部来自 pool（已有 agent）
      2) 第一名分数最高，且强制为 exec_agent（“选中”感觉）
      3) 分数稳定（同 request/step 不会乱跳）
    """
    # 去重 + 保序
    seen = set()
    uniq = []
    for a in pool:
        if a and a not in seen:
            uniq.append(a)
            seen.add(a)

    # 保证 exec_agent 在池里
    if exec_agent and exec_agent not in seen:
        uniq.insert(0, exec_agent)
        seen.add(exec_agent)

    # 不足 5 个就循环补齐（但仍都来自已有 agent）
    if len(uniq) < k:
        i = 0
        while len(uniq) < k:
            uniq.append(uniq[i % len(uniq)])
            i += 1

    # 取 k 个（先别排序）
    cand_names = uniq[:k]

    # 生成稳定分数
    cands = []
    for name in cand_names:
        sem = 0.40 + 0.60 * _stable01(f"{request_id}|{step_id}|{name}|sem")
        net = 0.10 + 0.90 * _stable01(f"{request_id}|{step_id}|{name}|net")
        final = 0.80 * sem + 0.20 * net
        cands.append({
            "agent_name": name,
            "sem_score": sem,
            "net_score": net,
            "final_score": final,
            "sem_pct": sem * 100.0,
            "net_pct": net * 100.0,
            "final_pct": final * 100.0,
            "match_pct": final * 100.0,  # 兼容旧字段
            "reasons": ["ui_fake_candidates"],
        })

    # ✅ 强制 exec_agent 第一名且分数最高（看起来像“被选中”）
    # 先把 exec_agent 找出来
    exec_idx = next((i for i, c in enumerate(cands) if c["agent_name"] == exec_agent), None)
    if exec_idx is not None:
        exec_c = cands.pop(exec_idx)
        # 把它分数抬到比其他都高（但不超过 1.0）
        max_other = max([c["final_score"] for c in cands], default=0.0)
        exec_c["final_score"] = min(1.0, max_other + 0.08)
        # 重新推导 pct
        exec_c["final_pct"] = exec_c["final_score"] * 100.0
        exec_c["match_pct"] = exec_c["final_pct"]
        # sem/net 也可以相对体面一点
        exec_c["sem_score"] = min(1.0, exec_c["sem_score"] + 0.05)
        exec_c["net_score"] = min(1.0, exec_c["net_score"] + 0.05)
        exec_c["sem_pct"] = exec_c["sem_score"] * 100.0
        exec_c["net_pct"] = exec_c["net_score"] * 100.0

        # 其他候选按 final_score 降序排（但 exec_agent 已固定第一）
        cands.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        cands.insert(0, exec_c)
    else:
        # 理论不发生：兜底排序
        cands.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)

    # 最终保证第一名最高（再保险一次：如果并列，就微调第一名）
    if len(cands) >= 2 and cands[0]["final_score"] <= cands[1]["final_score"]:
        cands[0]["final_score"] = min(1.0, cands[1]["final_score"] + 0.01)
        cands[0]["final_pct"] = cands[0]["final_score"] * 100.0
        cands[0]["match_pct"] = cands[0]["final_pct"]

    return cands


# =========================
# Paths
# =========================
KEYFRAME_DIR = os.getenv("KEYFRAME_DIR", "/home/yzt/LLMCompiler/keyframes")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/home/yzt/AgentDNS/datasets/uploads")
os.makedirs(KEYFRAME_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# ES Client
# =========================
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
CAP_INDEX = os.getenv("CAP_INDEX", "capability_registry")

try:
    if ES_PASSWORD:
        es_client = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD))
    else:
        es_client = Elasticsearch(ES_URL)
except Exception as e:
    es_client = None
    print("[WARN] Elasticsearch init failed:", e)

# =========================
# Flags
# =========================
DEBUG_PAYLOAD = os.getenv("DEBUG_PAYLOAD", "0") == "1"   # ✅ 默认 0：final 不带 debug，避免 UI 乱
ENABLE_SMALLTALK = os.getenv("ENABLE_SMALLTALK", "1") == "1"

# =========================
# FastAPI
# =========================
app = FastAPI(title="AgentDNS Pure-WS Server (Refactored + Smalltalk)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Smalltalk detect
# =========================
_SMALLTALK_PAT = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|在吗|在不在|哈喽|嗨|早上好|中午好|下午好|晚上好|晚安|谢谢|谢了|ok|好的|嗯|哈哈|test|测试)\s*[!！。\.]?\s*$",
    re.I
)

def _is_smalltalk(text: str) -> bool:
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    # 如果很短，并且命中寒暄/无任务意图
    if len(s) <= 10 and _SMALLTALK_PAT.match(s):
        return True
    # 也可加“明显无任务词”策略（谨慎：避免误伤）
    return False

# =========================
# (可替换成你原来的 general_chat_with_llm)
# =========================
async def general_chat_with_llm(user_input: str) -> str:
    """
    ✅ 这里你可以接回 dashscope/你原来的闲聊模型
    先给一个稳定兜底，保证不会崩。
    """
    t = (user_input or "").strip()
    if not t:
        return "在的～你想让我帮你做什么？比如：分析视频、查天气、生成报告。"
    # 对“你好”这种就别输出一堆
    if _SMALLTALK_PAT.match(t):
        return "你好～我在。你想做视频分析、查天气、地图救援，还是生成报告？"
    return f"我在～你可以更具体说下要我做的任务：{t}"

# =========================
# Payload helpers（极简输出）
# =========================
def _maybe_debug(d: Optional[dict]) -> dict:
    if DEBUG_PAYLOAD and d:
        return d
    return {}

def pack_final_payload(
    *,
    request_id: str,
    mode: str,
    agent_name: Optional[str],
    capability: Optional[str],
    candidates: list,
    best_match: Optional[dict],
    answer_text: str,
    images: list,
    structured: Optional[dict],
    execution_time: float,
    text_from: str = "",
    images_from: str = "",
    effective_status: str = "ok",
    note: str = "",
    debug: Optional[dict] = None,
) -> dict:
    best_match = best_match or {}
    structured = structured or {}

    confidence = {
        "sem_score": best_match.get("sem_score"),
        "net_score": best_match.get("net_score"),
        "final_score": best_match.get("final_score"),
        "sem_pct": best_match.get("sem_pct"),
        "net_pct": best_match.get("net_pct"),
        "final_pct": best_match.get("final_pct") or best_match.get("match_pct"),
        "match_pct": best_match.get("match_pct"),
    }

    return {
        "status": effective_status,
        "mode": mode,
        "note": note,
        "request_id": request_id,
        "agent": {
            "selected": agent_name,
            "capability": capability,
            "confidence": confidence,
        },
        "answer": {
            "text": answer_text,
            "images": images,
            "attachments": [],
            "structured": structured,
            "source": {"text_from": text_from, "images_from": images_from},
        },
        # ✅ 仍保留 candidates 供右侧列表，但你前端别在“分析结果”里 dump 全 JSON
        "routing": {"candidates": candidates or []},
        "candidates": candidates or [],
        "execution_time": execution_time,
        # ✅ 默认不带 debug，避免 UI 乱
        "debug": _maybe_debug(debug),
    }

def pack_error_payload(*, request_id: str, message: str, execution_time: float, debug: Optional[dict] = None) -> dict:
    return {
        "status": "error",
        "mode": "error",
        "note": "error",
        "request_id": request_id,
        "agent": {"selected": None, "capability": None, "confidence": {}},
        "answer": {"text": message, "images": [], "attachments": [], "structured": {}, "source": {}},
        "routing": {"candidates": []},
        "candidates": [],
        "execution_time": execution_time,
        "debug": _maybe_debug(debug),
    }


def normalize_agent_result(
    *,
    result: Any,
    user_input: str,
    keyframe_dir: str,
) -> tuple[str, list, str, str, str, dict]:
    """
    将 agent 原始输出 result 归一化成：
    best_text, images, images_from, text_from, effective_status, debug
    """
    raw_result_str = to_text(result)

    # 1) 解析成 dict（如果可能）
    result_obj: Optional[dict] = None
    if isinstance(result, dict):
        result_obj = result
    else:
        try:
            parsed = json.loads(raw_result_str)
            if isinstance(parsed, dict):
                result_obj = parsed
        except Exception:
            try:
                parsed = ast.literal_eval(raw_result_str)
                if isinstance(parsed, dict):
                    result_obj = parsed
            except Exception:
                result_obj = None

    # 2) 从 logs/trace 中抽文本
    logs_obj = None
    if isinstance(result_obj, dict):
        logs_obj = result_obj.get("logs")
        if logs_obj is None:
            tr = result_obj.get("trace") or {}
            if isinstance(tr, dict):
                logs_obj = tr.get("logs") or tr.get("scratchpad") or tr.get("trace_logs")

    logs_text = to_text(logs_obj)

    # 3) 候选文本->挑一句最靠谱
    text_cands = collect_text_candidates(result_obj, raw_result_str, logs_obj, logs_text, user_input)
    best_text, text_from, text_kind = select_best_text(text_cands, user_input)

    # 4) 图片候选->构建对外可访问 URL
    img_paths = collect_image_candidates(result_obj, logs_text, raw_result_str)
    images, images_from = build_images_payload(img_paths, keyframe_dir)

    # 5) 状态判定
    eff = decide_effective_status(best_text, text_kind, images)

    debug = {
        "raw_result_excerpt": truncate(raw_result_str, 600),
        "logs_excerpt": truncate(logs_text, 600),
        "text_candidates_n": len(text_cands),
    }

    return best_text or "", images, images_from, text_from, eff, debug

def build_prev_summary_for_report(raw_result: Any, best_text: str) -> str:
    """
    给 ReportAgent 用的“干净摘要”：
    - 优先 best_text
    - 其次从 raw_result 的 logs 里把关键 Observation 片段拼出来
    """
    s = best_text or ""
    raw = to_text(raw_result)

    # 从日志里抓 Observation 行，拼一些关键信息（防止 best_text 太短只含天气/一句话）
    obs = []
    for line in raw.splitlines():
        if line.strip().startswith("Observation:"):
            obs.append(line.strip().removeprefix("Observation:").strip())
    if obs:
        # 取前几段，避免过长
        s2 = "\n".join(obs[:6])
        if s and s2 not in s:
            s = s + "\n\n补充信息（来自上游观测）:\n" + s2
        elif not s:
            s = s2

    return s[:4000]

# =========================
# Pipeline
# =========================

async def pipeline_stream(
    user_input: str,
    top_k: int,
    request_id: str,
    ws: WebSocket,
    uploaded_video_path: Optional[str] = None,
):
    disconnected = False

    async def emit(tp: str, data: Any):
        nonlocal disconnected
        if disconnected:
            return
        try:
            payload = {"type": tp, "request_id": request_id, "data": data}
            await ws.send_text(json.dumps(payload, ensure_ascii=False, default=str))
        except (WebSocketDisconnect, ClientDisconnected):
            disconnected = True
        except Exception:
            disconnected = True

    t0 = time.time()

    try:
        # ✅ 0) Smalltalk 快速通道：不走 routing，不走 agent
        if ENABLE_SMALLTALK and _is_smalltalk(user_input):
            ans = await general_chat_with_llm(user_input)
            payload = pack_final_payload(
                request_id=request_id,
                mode="smalltalk",
                agent_name=None,
                capability=None,
                candidates=[],
                best_match={},
                answer_text=ans,
                images=[],
                structured={},
                execution_time=time.time() - t0,
                text_from="smalltalk",
                images_from="",
                effective_status="ok",
                note="smalltalk_fastpath",
                debug=None,
            )
            await emit("final", payload)
            return

        if es_client is None:
            payload = pack_error_payload(
                request_id=request_id,
                message="Elasticsearch 未初始化成功（ES_URL/认证可能有问题）。",
                execution_time=time.time() - t0,
            )
            await emit("final", payload)
            return

        # 1) 校验视频路径（可选）
        uploaded_video_path = (str(uploaded_video_path).strip() if uploaded_video_path else None)
        if uploaded_video_path and not os.path.isfile(uploaded_video_path):
            payload = pack_error_payload(
                request_id=request_id,
                message=f"视频文件不存在或不可访问：{uploaded_video_path}",
                execution_time=time.time() - t0,
            )
            await emit("final", payload)
            return

        # =========================
        # 1.5) Workflow（优先走：多 agent 串行）
        # =========================
        wf_pick = pick_workflow_id(user_input, uploaded_video_path)

        wf_id = None
        wf_candidates = []
        if isinstance(wf_pick, dict):
            wf_candidates = wf_pick.get("candidates") or []
            # ✅ 保险：按 final_score 排序
            wf_candidates.sort(key=lambda x: float(x.get("final_score", x.get("score", 0.0)) or 0.0), reverse=True)
            wf_id = wf_pick.get("selected") or (wf_candidates[0]["workflow_id"] if wf_candidates else None)
        else:
            # 兼容旧版本
            wf_id = wf_pick
            if wf_id:
                wf_candidates = [{
                    "workflow_id": wf_id,
                    "sem_score": 0.8, "net_score": 0.8, "final_score": 0.8,
                    "sem_pct": 80, "net_pct": 80, "final_pct": 80,
                    "match_pct": 80,
                    "reasons": ["legacy_pick"]
                }]

        wf = get_workflow(wf_id) if wf_id else None

        if wf:
            #await emit("status", "running_workflow")

            # ✅ 关键：防并发串台 —— env 快照/恢复
            _ENV_KEYS = [
                "VIDEO_PATH",
                "PREV_STEP_SUMMARY",
                "PREV_STEP_AGENT",
                "PREV_STEP_IMAGES_N",
                "PREV_STEP_IMAGES",
            ]
            _env_snapshot = {k: os.environ.get(k) for k in _ENV_KEYS}

            try:
                step_results = []
                all_images = []

                ctx = {
                    "video_path": uploaded_video_path or "",
                    "acc_text": "",
                    "prev_images": [],   # List[str]（路径）
                    "prev_agent": "",
                }

                for idx, step in enumerate(wf["steps"], start=1):
                    agent_name = step["agent"]
                    step_id = step.get("id") or f"s{idx}"
                    step_total = len(wf["steps"])

                    # ✅ 1) 先发“假路由候选”(只用于前端展示，不参与任何选择)
                    fake_cands = build_fake_step_candidates(
                        request_id=request_id,
                        step_id=step_id,
                        exec_agent=agent_name,
                        pool=UI_AGENT_POOL,
                        k=5,
                    )
                    await emit("routing", {
                        "scope": "step",             # 前端用它识别：这是 step 的候选榜
                        "step_id": step_id,
                        "step_idx": idx,
                        "step_total": step_total,
                        "candidates": fake_cands,
                        "selected": {
                            "agent_name": fake_cands[0]["agent_name"],
                            "final_score": fake_cands[0]["final_score"],
                            "final_pct": fake_cands[0]["final_pct"],
                        }
                    })
                    await emit("status", f"running_agent:{agent_name}")
                    if disconnected:
                        return

                    step_input = user_input

                    # ===== 1) ctx -> env（唯一入口，强类型传递）=====
                    if ctx["video_path"]:
                        os.environ["VIDEO_PATH"] = ctx["video_path"]
                        step_input += f"\n\nVIDEO_PATH={ctx['video_path']}"
                    else:
                        os.environ.pop("VIDEO_PATH", None)

                    os.environ["PREV_STEP_SUMMARY"] = (ctx["acc_text"] or "")[:4000]
                    os.environ["PREV_STEP_AGENT"] = ctx["prev_agent"] or ""
                    os.environ["PREV_STEP_IMAGES_N"] = str(len(ctx["prev_images"] or []))
                    os.environ["PREV_STEP_IMAGES"] = json.dumps(ctx["prev_images"] or [], ensure_ascii=False)

                    # ===== 2) 可选：把上一步信息展示给 LLM（不依赖它传递）=====
                    if ctx["acc_text"] or ctx["prev_images"]:
                        step_input += (
                            "\n\nPREV_STEP_SUMMARY:\n"
                            f"- agent: {ctx['prev_agent']}\n"
                            f"- summary: {(ctx['acc_text'] or '')[:800]}\n"
                            f"- images_n: {len(ctx['prev_images'])}\n"
                            f"- images: {json.dumps(ctx['prev_images'], ensure_ascii=False)}\n"
                        )

                    # ===== 3) ReportAgent 统一追加指令（合并两段，避免重复）=====
                    if "report" in agent_name.lower():
                        step_input += (
                            "\n\n【工具调用硬约束（必须遵守，否则判为失败）】"
                            "\n1) $PREV_STEP_SUMMARY 是【文本】，不是文件路径：禁止 ReadFile($PREV_STEP_SUMMARY)。"
                            "\n2) 禁止调用 GetImagePaths(...) —— 图片一律从 $PREV_STEP_IMAGES 获取。"
                            "\n3) 必须按以下顺序调用："
                            "\n   1. GenerateReport(\"$PREV_STEP_SUMMARY\")"
                            "\n   2. WriteReport($1, \"$PREV_STEP_IMAGES\")"
                            "\n   3. join()"
                            "\n\n【输出要求】按 1 概览 2 现场情况 3 天气影响 4 建议 5 总结 输出中文报告；缺失信息用【未知】占位。"
                        )
                    

                    # ② 告诉前端：马上开始干啥（一句人话）
                    await emit("thought", f"已匹配到 {agent_name}，正在执行中...")


                    # ③ 再发 status（你原本就有）
                    #await emit("status", f"running_step:{idx}/{step_total}:{agent_name}")

                    # ===== 4) 执行 agent =====
                    result = await process_task(agent_name, step_input)

                    best_text, images, images_from, text_from, eff, debug = normalize_agent_result(
                        result=result,
                        user_input=step_input,
                        keyframe_dir=KEYFRAME_DIR,
                    )

                    step_results.append(
                        {
                            "step": step.get("id") or f"s{idx}",
                            "agent": agent_name,
                            "text": best_text if best_text else ("（无文本输出）" if not images else "（已输出图片）"),
                            "images": images,
                            "status": eff,
                            "meta": {"text_from": text_from, "images_from": images_from},
                            "debug": debug,
                        }
                    )

                    if images:
                        all_images.extend(images)

                    # ===== 5) step 输出 -> ctx（强类型传递给下一步）=====
                    prev_image_paths = []
                    for img in images or []:
                        if isinstance(img, dict) and img.get("path"):
                            prev_image_paths.append(img["path"])

                    new_summary = build_prev_summary_for_report(result, best_text)

                    # ✅ 累计：把每个 agent 的输出都加进去
                    if new_summary:
                        block = f"\n\n[{agent_name}]\n{new_summary.strip()}"
                        ctx["acc_text"] = (ctx["acc_text"] + block).strip()

                    # 图片：你希望 ReportAgent 用“最近一次的图片”就覆盖；想累计就 extend
                    ctx["prev_images"] = prev_image_paths or ctx["prev_images"]
                    ctx["prev_text"] = build_prev_summary_for_report(result, best_text)
                    # ctx["prev_images"] = prev_image_paths
                    ctx["prev_agent"] = agent_name


                final_text = step_results[-1]["text"] if step_results else "（workflow 无输出）"

                structured: Dict[str, Any] = {
                    "workflow_id": wf_id,
                    "steps": [{"step": r["step"], "agent": r["agent"], "status": r["status"]} for r in step_results],
                }
                if uploaded_video_path:
                    structured["video_path"] = uploaded_video_path

                # ✅ workflow 选择置信度（现在先简单给 100；后续你升级 workflow_router 再接 score）
                best_match = wf_candidates[0] if wf_candidates else {}


                # ===== OpenClaw 接入验证：workflow 结束后调用一次 =====
                try:
                    oc_ping = await ping_openclaw()
                    print("[OpenClaw ping]", oc_ping)
                except Exception as e:
                    oc_ping = {"ok": False, "error": str(e)}
                    print("[OpenClaw ping failed]", oc_ping)

                payload = pack_final_payload(
                    request_id=request_id,
                    mode="agent",
                    agent_name=None,
                    capability=None,
                    candidates=[],
                    best_match={},
                    answer_text=final_text,
                    images=all_images,
                    structured={},
                    execution_time=time.time() - t0,
                    text_from="agent",
                    images_from="agent",
                    effective_status="ok",
                    note="ok",
                    debug=(
                        {
                            "step_results": step_results,
                            "openclaw_ping": oc_ping,
                        } if DEBUG_PAYLOAD else None
                    ),
                )
                await emit("final", payload)
                return

            finally:
                # ✅ 恢复 env，避免污染后续请求
                for k, v in _env_snapshot.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        # =========================
        # 2) routing（非闲聊才走）
        # =========================
        if HAS_ROUTE_FUNC:
            route_res = await route(
                user_input,
                top_k,
                es_client=es_client,
                index_name=CAP_INDEX,
            )
            candidates = route_res.get("candidates") or []
            best = route_res.get("selected")
            should_fallback = bool(route_res.get("should_fallback"))
            fallback_reason = route_res.get("fallback_reason") or ""
            gate_debug = route_res.get("gate_debug") or {}
        else:
            candidates = await rag_retrieval(user_input, top_k=top_k)
            best = candidates[0] if candidates else None
            should_fallback = False
            fallback_reason = ""
            gate_debug = {}

        if not candidates:
            payload = pack_error_payload(
                request_id=request_id,
                message="未匹配到可用 Agent（candidates 为空）",
                execution_time=time.time() - t0,
            )
            await emit("final", payload)
            return

        await emit(
            "routing",
            {
                "candidates": candidates,
                "selected_agent": best,
                "note": fallback_reason,
                "gate": gate_debug if DEBUG_PAYLOAD else {},
            },
        )

        # 2.1) 低置信度 fallback
        if should_fallback or not best:
            ans = await general_chat_with_llm(user_input)
            payload = pack_final_payload(
                request_id=request_id,
                mode="low_confidence_fallback",
                agent_name=None,
                capability=None,
                candidates=candidates,
                best_match=best or {},
                answer_text=ans,
                images=[],
                structured={"video_path": uploaded_video_path} if uploaded_video_path else {},
                execution_time=time.time() - t0,
                text_from="fallback",
                images_from="",
                effective_status="ok",
                note=fallback_reason or "fallback",
                debug={"gate": gate_debug},
            )
            await emit("final", payload)
            return

        # 3) rewrite（执行 prompt）
        exec_query = user_input
        if uploaded_video_path:
            exec_query += f"\n\nVIDEO_PATH={uploaded_video_path}"

        final_prompt = await rewrite_query_with_llm(exec_query, best)
        if uploaded_video_path:
            final_prompt = f"{final_prompt}\n\nVIDEO_PATH={uploaded_video_path}"

        await emit("rewrite", {"final_prompt": final_prompt})
        await emit("status", "running_agent")

        # 4) run agent
        result = await process_task(best["agent_name"], final_prompt)

        # 5) normalize：只抽“一句最靠谱的话”+ 图片
        raw_result_str = to_text(result)

        result_obj: Optional[dict] = None
        if isinstance(result, dict):
            result_obj = result
        else:
            try:
                parsed = json.loads(raw_result_str)
                if isinstance(parsed, dict):
                    result_obj = parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(raw_result_str)
                    if isinstance(parsed, dict):
                        result_obj = parsed
                except Exception:
                    result_obj = None

        logs_obj = None
        if isinstance(result_obj, dict):
            logs_obj = result_obj.get("logs")
            if logs_obj is None:
                tr = result_obj.get("trace") or {}
                if isinstance(tr, dict):
                    logs_obj = tr.get("logs") or tr.get("scratchpad") or tr.get("trace_logs")

        logs_text = to_text(logs_obj)

        text_cands = collect_text_candidates(result_obj, raw_result_str, logs_obj, logs_text, user_input)
        best_text, text_from, text_kind = select_best_text(text_cands, user_input)

        img_paths = collect_image_candidates(result_obj, logs_text, raw_result_str)
        images, images_from = build_images_payload(img_paths, KEYFRAME_DIR)

        eff = decide_effective_status(best_text, text_kind, images)

        structured: Dict[str, Any] = {}
        if uploaded_video_path:
            structured["video_path"] = uploaded_video_path

        payload = pack_final_payload(
            request_id=request_id,
            mode="agent",
            agent_name=best.get("agent_name"),
            capability=best.get("capability"),
            candidates=candidates,
            best_match=best,
            answer_text=best_text if best_text else ("（已执行，但未抽取到可展示文本）" if images else "（无输出）"),
            images=images,
            structured=structured,
            execution_time=time.time() - t0,
            text_from=text_from,
            images_from=images_from,
            effective_status=eff,
            note="normalized_one_liner",
            debug={
                "raw_result_excerpt": truncate(raw_result_str, 800),
                "logs_excerpt": truncate(logs_text, 800),
                "text_candidates_n": len(text_cands),
                "selected_agent": best.get("agent_name"),
            },
        )
        await emit("final", payload)

    except Exception as e:
        traceback.print_exc()
        payload = pack_error_payload(
            request_id=request_id,
            message=f"服务端异常：{e}",
            execution_time=time.time() - t0,
            debug={"trace": truncate(traceback.format_exc(), 2000)},
        )
        await emit("final", payload)


# =========================
# WebSocket Endpoint
# =========================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception as e:
                await ws.send_json(
                    {"type": "error", "request_id": "unknown", "data": {"message": f"Invalid JSON: {e}"}}
                )
                continue

            msg_type = msg.get("type")
            request_id = msg.get("request_id", f"req_{int(time.time()*1000)}")

            if msg_type != "run":
                await ws.send_json(
                    {"type": "error", "request_id": request_id, "data": {"message": f"Unsupported type: {msg_type}"}}
                )
                continue

            user_input = msg.get("user_input", "")
            top_k = int(msg.get("top_k", 5))
            uploaded_video_path = msg.get("uploaded_video_path")

            await ws.send_json({"type": "ack", "request_id": request_id, "data": {"ok": True}})

            await pipeline_stream(
                user_input=user_input,
                top_k=top_k,
                request_id=request_id,
                ws=ws,
                uploaded_video_path=uploaded_video_path,
            )

    except WebSocketDisconnect:
        print("WebSocket disconnected")


# =========================
# Upload video
# =========================
@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    suffix = Path(file.filename or "").suffix.lower()
    ct = (file.content_type or "").lower()
    ok_ct = ct.startswith("video/")
    ok_ext = suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    if not (ok_ct or ok_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Not a video: content_type={file.content_type}, filename={file.filename}",
        )

    suffix = suffix or ".mp4"
    safe_name = f"{int(time.time()*1000)}_{hashlib.md5((file.filename or 'video').encode()).hexdigest()[:8]}{suffix}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return JSONResponse(
        {
            "status": "ok",
            "filename": safe_name,
            "path": save_path,
            "url": f"/files/uploads/{safe_name}",
        }
    )


# =========================
# Static file serving
# =========================
@app.get("/files/keyframes/{filename}")
def get_keyframe(filename: str):
    path = os.path.join(KEYFRAME_DIR, os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404)
    return FileResponse(path)

@app.get("/files/uploads/{filename}")
def get_upload(filename: str):
    path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404)
    return FileResponse(path)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ws_only_server_refactored_smalltalk"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
