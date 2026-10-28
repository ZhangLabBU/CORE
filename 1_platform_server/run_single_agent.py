import asyncio
import json
import argparse
from typing import Dict, Any, Optional
import time
import importlib

# 导入必要的模块
# from configs.VideoAgent.configs import CONFIGS as VIDEO_CONFIGS
# from configs.VideoAgent.tools import tools as video_tools
# from configs.KeyframeAgent.configs import CONFIGS as KEYFRAME_CONFIGS
# from configs.KeyframeAgent.tools import tools as keyframe_tools
# from configs.MapAgent.configs import CONFIGS as MAP_CONFIGS
# from configs.MapAgent.tools import tools as map_tools
# from configs.MeteorologyAgent.configs import CONFIGS as METEOROLOGY_CONFIGS
# from configs.MeteorologyAgent.tools import tools as meteorology_tools
# from configs.ReportAgent.configs import CONFIGS as REPORT_CONFIGS
# from configs.ReportAgent.tools import tools as report_tools

from src.utils.logger_utils import GlobalLogManager
from src.callbacks.callbacks import StatsCallbackHandler
from src.llm_compiler.constants import END_OF_PLAN
from src.llm_compiler.llm_compiler import LLMCompiler
from src.react.base import initialize_react_agent_executor
from src.utils.evaluation_utils import arun_and_time, normalize_answer
from src.utils.model_utils import get_model

import atexit
import re, ast, json
from typing import List, Dict, Tuple, Any


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
def _scratchpad_text(logs):
    if isinstance(logs, list):
        parts=[]
        for x in logs:
            if isinstance(x, dict):
                sp = x.get("Agent scratchpad")
                if isinstance(sp, str) and sp.strip():
                    parts.append(sp)
        if parts:
            return "\n\n".join(parts)
    return ""

def _parse_tool_observations(text: str) -> Dict[str, List[str]]:
    """
    text 这里我们会传入 scratchpad（不是整个 logs 的 json）
    """
    text = _to_text(text).replace("\\n", "\n")
    pat = re.compile(
        r"(?P<tool>[A-Za-z_]\w*)\([^\)]*\)\s*\nObservation:\s*(?P<obs>.*?)(?=\n[A-Za-z_]\w*\(|\Z)",
        flags=re.DOTALL,
    )
    out: Dict[str, List[str]] = {}
    for m in pat.finditer(text):
        tool = m.group("tool").strip()
        obs = m.group("obs").strip()
        if obs:
            out.setdefault(tool, []).append(obs)
    return out


def _extract_finish_balanced(text: str) -> str:
    t = _to_text(text).replace("\\n", "\n")
    pos = t.rfind("Action: Finish(")
    if pos == -1:
        return ""
    s = t[pos + len("Action: Finish(") :]
    depth = 1
    out = []
    in_quote = None
    esc = False
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
                out.pop()
                break
    payload = "".join(out).strip()
    if (payload.startswith('"') and payload.endswith('"')) or (
        payload.startswith("'") and payload.endswith("'")
    ):
        payload = payload[1:-1].strip()
    return payload


def _extract_write_report(text: str) -> Tuple[str, List[str]]:
    t = _to_text(text).replace("\\n", "\n")
    idx = t.rfind("WriteReport(")
    if idx == -1:
        return "", []
    tail = t[idx:]
    m = re.search(r"WriteReport\((.*),\s*(\[[^\]]*\])\s*\)\s*$", tail, flags=re.DOTALL)
    if not m:
        m2 = re.search(r"WriteReport\((.*)\)\s*$", tail, flags=re.DOTALL)
        return (m2.group(1).strip(), []) if m2 else ("", [])
    report = m.group(1).strip()
    paths_raw = m.group(2).strip()
    paths: List[str] = []
    try:
        parsed = ast.literal_eval(paths_raw)
        if isinstance(parsed, list):
            paths = [p for p in parsed if isinstance(p, str)]
    except Exception:
        paths = re.findall(r"(/[^ \n\t]+?\.(?:jpg|jpeg|png))", paths_raw, flags=re.I)
    # 去重
    uniq, seen = [], set()
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return report, uniq


def _extract_images_from_logs(logs: Any) -> List[str]:
    # ✅ 只用 scratchpad 文本
    text = _scratchpad_text(logs).replace("\\n", "\n")

    # 1) WriteReport(..., [paths]) 仍支持：直接从 text 里找
    _, report_paths = _extract_write_report(text)
    if report_paths:
        return report_paths  # 不截断，不 [:5]

    tools = _parse_tool_observations(text)
    paths: List[str] = []

    if "KeyframeExtractor" in tools:
        for obs in tools["KeyframeExtractor"]:
            paths += re.findall(r"(/[^ \n\t]+?\.(?:jpg|jpeg|png))", obs, flags=re.I)

    if not paths and "GetImagePaths" in tools:
        for obs in tools["GetImagePaths"]:
            try:
                parsed = ast.literal_eval(obs.strip())
                if isinstance(parsed, list):
                    paths += [p for p in parsed if isinstance(p, str)]
            except Exception:
                pass

    if not paths:
        paths = re.findall(r"(/[^ \n\t]+?\.(?:jpg|jpeg|png))", text, flags=re.I)

    # 去重保序（不截断）
    uniq, seen = [], set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def normalize_agent_output(raw_answer: Any, logs: Any) -> Dict[str, Any]:
    """
    ✅ 统一固定输出模板（所有 Agent 一致）
    ✅ 不省略（不截断，不加 …）
    ✅ 不再混入 ", "Question": ...（只解析 Agent scratchpad）
    """
    scratch = _scratchpad_text(logs)              # ✅ 核心：只取 scratchpad
    tools = _parse_tool_observations(scratch)     # ✅ 只对 scratchpad 做解析

    # -------- 1) 结论 --------
    conclusion = ""

    finish = _extract_finish_balanced(_to_text(raw_answer))
    if finish.strip():
        conclusion = finish.strip()
    else:
        ra = _to_text(raw_answer).replace("\\n", "\n").strip()
        if ra and ("Thought:" not in ra) and ("Action:" not in ra):
            conclusion = ra

    # 兜底：最后一个 Observation
    all_obs = [o for lst in tools.values() for o in lst if o.strip()]
    if not conclusion and all_obs:
        conclusion = all_obs[-1].strip()

    if not conclusion:
        conclusion = "已完成任务，但未提取到明确结论。"

    # -------- 2) 关键信息：不截断 --------
    key_points: List[str] = []
    for tool, lst in tools.items():
        if not lst:
            continue
        s = lst[-1].strip()
        if s:
            key_points.append(s)
    key_points = key_points[:6]   # 条数你可以自己调大

  
    # -------- 4) 图片：不截断 --------
    image_paths = _extract_images_from_logs(logs)

    # -------- 5) 固定模板输出 --------
    text_lines = []
    text_lines.append("【结论】")
    text_lines.append(conclusion)
    text_lines.append("")
    text_lines.append("【关键信息】")
    if key_points:
        text_lines += [f"- {p}" for p in key_points]
    else:
        text_lines.append("- （无）")
    text_lines.append("")
    # text_lines.append("【依据/来源】")
    # if evidence:
    #     text_lines += [f"- {e}" for e in evidence]
    # else:
    #     text_lines.append("- （无）")
    # text_lines.append("")
    text_lines.append("【附加内容】")
    text_lines.append(f"- 图片: {len(image_paths)} 张")

    fixed_text = "\n".join(text_lines)

    return {
        "display": {"text": fixed_text},
        "structured": {"images": [{"path": p} for p in image_paths]},
    }


# 注册退出时的钩子
atexit.register(GlobalLogManager.save_logs_to_file)

# 定义各Agent的输入文件路径
agent_task_files = {
    "KeyframeAgent": "datasets/keyframeagent_test.json",
    "VideoAgent": "datasets/videoagent_test.json",
    "MapAgent": "datasets/mapagent_test.json",
    "MeteorologyAgent": "datasets/meteorologyagent_test.json",
    "ReportAgent": "datasets/reportagent_test.json",
}

import os

os.makedirs("tasks", exist_ok=True)


def write_agent_tasks(agent_name: str, prompt: str) -> None:
    """将任务数据写入子Agent输入文件（覆盖当前任务数据）"""
    if agent_name not in agent_task_files:
        raise ValueError(f"Unknown agent name: {agent_name}")

    task_data = [{"id": 0, "level": "default", "question": prompt, "answer": ""}]

    filename = agent_task_files[agent_name]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(task_data, f, ensure_ascii=False, indent=4)
# =========================
# Dynamic loaders (IMPORTANT)
# =========================
AGENT_CONFIG_MODULE = {
    "KeyframeAgent": "configs.KeyframeAgent.configs",
    "VideoAgent": "configs.VideoAgent.configs",
    "MapAgent": "configs.MapAgent.configs",
    "MeteorologyAgent": "configs.MeteorologyAgent.configs",
    "ReportAgent": "configs.ReportAgent.configs",
}

AGENT_TOOLS_MODULE = {
    "KeyframeAgent": "configs.KeyframeAgent.tools",
    "VideoAgent": "configs.VideoAgent.tools",
    "MapAgent": "configs.MapAgent.tools",
    "MeteorologyAgent": "configs.MeteorologyAgent.tools",
    "ReportAgent": "configs.ReportAgent.tools",
}

def load_agent_configs(agent_name: str) -> Dict[str, Any]:
    mod_path = AGENT_CONFIG_MODULE.get(agent_name)
    if not mod_path:
        raise ValueError(f"Unknown agent: {agent_name}")
    mod = importlib.import_module(mod_path)
    cfg = getattr(mod, "CONFIGS", None)
    if cfg is None:
        raise ValueError(f"{mod_path} has no variable `CONFIGS`")
    return cfg

def load_agent_tools(agent_name: str):
    mod_path = AGENT_TOOLS_MODULE.get(agent_name)
    if not mod_path:
        raise ValueError(f"Unknown agent: {agent_name}")
    mod = importlib.import_module(mod_path)
    tools = getattr(mod, "tools", None)
    if tools is None:
        raise ValueError(f"{mod_path} has no variable `tools`")
    print(f"[LOADED TOOLS] {mod_path} names =", [t.name for t in tools])
    return tools


async def run_single_agent(
    agent_name: str,
    prompt: str,
    model_type: str = "openai",
    model_name: Optional[str] = None,
    use_react: bool = True,
    vllm_port: Optional[int] = None,
) -> Dict[str, Any]:
    """
    运行指定的单个agent

    Args:
        agent_name: 要运行的agent名称
        prompt: 输入提示
        model_type: 模型类型
        model_name: 模型名称（可选）
        use_react: 是否使用ReAct模式
        vllm_port: vllm端口（可选）

    Returns:
        包含执行结果的字典
    """
    # 根据agent_name获取配置和工具
        # ✅ 动态加载（用哪个 agent 才 import 哪个 tools/configs）
    if agent_name not in AGENT_CONFIG_MODULE:
        raise ValueError(f"Unknown agent name: {agent_name}")

    configs = load_agent_configs(agent_name)
    tools = load_agent_tools(agent_name)


    # 获取模型名称
    if model_name is None:
        model_name = configs.get("default_model", "gpt-3.5-turbo")

    # 确定prompt类型
    if model_type in ["openai", "azure"]:
        prompt_type = "gpt"
    else:
        prompt_type = "llama"

    # 初始化agent
    if use_react:
        print(f"运行 ReAct 模式: {agent_name}")
        if "prompt" not in configs:
            raise ValueError(f"React config requires a prompt for {agent_name}")

        prompt_template = configs["prompt"][prompt_type]

        llm = get_model(
            model_type=model_type,
            model_name=model_name,
            vllm_port=vllm_port,
            stream=False,
            temperature=0,
        )

        # 初始化agent执行器
        agent = initialize_react_agent_executor(
            name=f"{agent_name}_React",
            llm=llm,
            tools=tools,
            prompt=prompt_template,
            verbose=True,
        )
    else:
        print(f"运行 LLM Compiler 模式: {agent_name}")

        if "prompts" not in configs:
            raise ValueError(f"LLMCompiler config requires prompts for {agent_name}")

        llm = get_model(
            model_type=model_type,
            model_name=model_name,
            vllm_port=vllm_port,
            stream=False,
            temperature=0,
        )

        planner_llm = get_model(
            model_type=model_type,
            model_name=model_name,
            vllm_port=vllm_port,
            stream=False,
            temperature=0,
        )

        prompts = configs["prompts"][prompt_type]

        agent = LLMCompiler(
            name=f"{agent_name}_LLMCompiler",
            tools=tools,
            planner_llm=planner_llm,
            planner_example_prompt=prompts["planner_prompt"],
            planner_example_prompt_replan=prompts.get("planner_prompt_replan"),
            planner_stop=[END_OF_PLAN],
            planner_stream=False,
            agent_llm=llm,
            joinner_prompt=prompts["output_prompt"],
            joinner_prompt_final=prompts.get("output_prompt_final"),
            max_replans=configs.get("max_replans", 1),
            benchmark=False,
        )

    # 运行agent
    try:
        # 先清空之前的日志
        GlobalLogManager.clear_logs()

        # 运行agent - 这里直接调用，让原有的日志机制工作
        start_time = time.time()

        if use_react:
            # 对于ReAct模式，直接运行
            raw_answer = await agent.arun(prompt)
        else:
            # 对于LLMCompiler模式，直接运行
            raw_answer = await agent.arun(prompt)

        end_time = time.time()
        e2e_time = end_time - start_time

        # 获取日志 - 直接从GlobalLogManager获取
        logs = (
            GlobalLogManager._log_data if hasattr(GlobalLogManager, "_log_data") else []
        )

        # 构建返回结果
        # normalized = normalize_agent_output(raw_answer, logs)

        # result = {
        #     "agent_name": agent_name,
        #     "status": "success",
        #     "execution_time": e2e_time,
        #     "question": prompt,
        #     # ✅ 统一给前端用的
        #     "display": normalized["display"],
        #     "structured": normalized["structured"],
        #     # ✅ 保留原始信息给你排查
        #     "trace": {
        #         "raw_answer": raw_answer,
        #         "normalized_answer": normalize_answer(raw_answer),
        #         "logs": logs,
        #     },
        # }
        result = {
            "agent_name": agent_name,
            "status": "success",
            "execution_time": e2e_time,
            "question": prompt,

            # 原材料
            "raw_answer": raw_answer,
            "logs": logs,  # 原始 logs（list[dict]）
            "images": [],  # 可选：你若能从工具里提取就填，提取逻辑也可以放 task_processing

            # 调试信息
            "trace": {
                "normalized_answer": normalize_answer(raw_answer),
            },
        }

        return result

    except Exception as e:
        import traceback

        traceback.print_exc()

        return {
            "agent_name": agent_name,
            "status": "error",
            "error": str(e),
            "question": prompt,
            "execution_time": None,
            "raw_answer": None,
        }


async def process_agent_request(
    agent_name: str,
    prompt: str,
    model_type: str = "openai",
    model_name: Optional[str] = None,
    use_react: bool = True,
    vllm_port: Optional[int] = None,
) -> Dict[str, Any]:
    """
    处理agent请求的入口函数

    Args:
        agent_name: agent名称
        prompt: 输入提示
        model_type: 模型类型
        model_name: 模型名称（可选）
        use_react: 是否使用ReAct模式
        vllm_port: vllm端口（可选）

    Returns:
        执行结果
    """
    # 写入任务文件
    write_agent_tasks(agent_name, prompt)

    # 运行agent
    result = await run_single_agent(
        agent_name=agent_name,
        prompt=prompt,
        model_type=model_type,
        model_name=model_name,
        use_react=use_react,
        vllm_port=vllm_port,
    )

    # 保存日志到文件
    GlobalLogManager.save_logs_to_file()

    # 同时保存到所有任务日志文件
    try:
        execution_log = {
            "agent_name": agent_name,
            "question": prompt,
            "raw_answer": result.get("raw_answer"),
            "execution_time": result.get("execution_time"),
            "status": result.get("status", "unknown"),
        }
        if result.get("status") == "error":
            execution_log["error"] = result.get("error")
        GlobalLogManager.save_all_tasks_logs(execution_log)
    except Exception as e:
        print(f"保存到所有任务日志文件时出错: {e}")

    # 返回结果
    return result


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="运行单个Agent")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=[
            "KeyframeAgent",
            "VideoAgent",
            "MapAgent",
            "MeteorologyAgent",
            "ReportAgent",
        ],
        help="要运行的Agent名称",
    )
    parser.add_argument("--prompt", type=str, required=True, help="任务提示语")
    parser.add_argument(
        "--model_type",
        type=str,
        default="openai",
        choices=["openai", "vllm", "azure", "friendli"],
        help="模型类型",
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="模型名称（覆盖默认值）"
    )
    parser.add_argument(
        "--use_react",
        action="store_true",
        default=True,
        help="使用ReAct模式（默认True）",
    )
    parser.add_argument(
        "--no_react",
        action="store_false",
        dest="use_react",
        help="不使用ReAct模式（使用LLMCompiler）",
    )
    parser.add_argument("--vllm_port", type=int, default=None, help="vllm端口")

    args = parser.parse_args()

    # 运行agent
    result = asyncio.run(
        process_agent_request(
            agent_name=args.agent,
            prompt=args.prompt,
            model_type=args.model_type,
            model_name=args.model_name,
            use_react=args.use_react,
            vllm_port=args.vllm_port,
        )
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("Agent 执行结果:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["status"] == "success":
        print("\n" + "=" * 60)
        print("主要输出:")
        print("=" * 60)
        print(f"原始回答: {result['raw_answer']}")
        print(f"执行时间: {result['execution_time']:.2f}秒")

        # 打印详细的日志
        print("\n" + "=" * 60)
        print("详细执行日志:")
        print("=" * 60)
        for log_entry in result.get("logs", []):
            print(json.dumps(log_entry, indent=2, ensure_ascii=False))
            print("-" * 40)


if __name__ == "__main__":
    main()
