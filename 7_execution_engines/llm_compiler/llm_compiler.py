import asyncio
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast, Tuple

from pydantic.v1 import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.llms.base import BaseLLM
from langchain.prompts.base import StringPromptValue

from src.callbacks.callbacks import AsyncStatsCallbackHandler
from src.chains.chain import Chain
from src.llm_compiler.constants import JOINNER_REPLAN
from src.llm_compiler.planner import Planner
from src.llm_compiler.task_fetching_unit import Task, TaskFetchingUnit
from src.tools.base import StructuredTool, Tool
from src.utils.logger_utils import log
from src.utils.logger_utils import GlobalLogManager
import json

class LLMCompilerAgent:
    """Self defined agent for LLM Compiler."""

    def __init__(self, llm: BaseLLM) -> None:
        # 初始化函数，接收一个BaseLLM类型的语言模型实例
        self.llm = llm

    async def arun(self, prompt: str, callbacks=None) -> str:
        """
        异步运行方法，用于根据给定的提示生成响应。

        Args:
            prompt (str): 要生成响应的提示文本。
            callbacks: 可选的回调函数，用于在生成过程中进行一些额外的操作。

        Returns:
            str: 生成的响应文本。
        """
        response = await self.llm.agenerate_prompt(
            prompts=[StringPromptValue(text=prompt)],
            stop=["<END_OF_RESPONSE>"],
            callbacks=callbacks,
        )
        if isinstance(self.llm, BaseChatModel):
            return response.generations[0][0].message.content

        if isinstance(self.llm, BaseLLM):
            return response.generations[0][0].text

        raise ValueError("LLM must be either BaseChatModel or BaseLLM")

# 定义LLMCompiler类，它是LLMCompiler Engine类，继承自Chain类
class LLMCompiler(Chain, extra="allow"):
    """LLMCompiler Engine."""

    """The step container to use."""
    input_key: str = "input"
    output_key: str = "output"
    # name: str = Field(..., description="The name of the LLM compiler")

    def __init__(
        self,
        name: str,  # 新增 `name` 参数
        tools: Sequence[Union[Tool, StructuredTool]],
        planner_llm: BaseLLM,
        planner_example_prompt: str,
        planner_example_prompt_replan: Optional[str],
        planner_stop: Optional[list[str]],
        planner_stream: bool,
        agent_llm: BaseLLM,
        joinner_prompt: str,
        joinner_prompt_final: Optional[str],
        max_replans: int,
        benchmark: bool,
        **kwargs,
    ) -> None:
        """
        初始化函数，用于设置LLMCompiler的各种参数。

        Args:
            tools (Sequence[Union[Tool, StructuredTool]]): 要使用的工具序列，可以是Tool或StructuredTool类型。
            max_replans (int): 最大重新规划次数。
            benchmark (bool): 是否收集基准统计信息。

        Planner Args:
            planner_llm (BaseLLM): 用于规划的语言模型。
            planner_example_prompt (str): 规划的示例提示。
            planner_example_prompt_replan (Optional[str]): 重新规划的示例提示，如果未指定，将使用planner_example_prompt。
            planner_stop (Optional[list[str]]): 规划的停止标记列表。
            planner_stream (bool): 是否流式传输规划过程。

        Agent Args:
            agent_llm (BaseLLM): 用于代理的语言模型。
            joinner_prompt (str): 用于连接操作的提示。
            joinner_prompt_final (Optional[str]): 在最后一次重新规划迭代中使用的连接提示，如果未指定，将使用joinner_prompt。
        """
        super().__init__(name=name,**kwargs)
        self.log_data: Dict[str, Any] = {}  # 初始化一个空的JSON对象（字典）

        if not planner_example_prompt_replan:
            log(
                "Replan example prompt not specified, using the same prompt as the planner."
            )
            planner_example_prompt_replan = planner_example_prompt

        self.planner = Planner(
            llm=planner_llm,
            example_prompt=planner_example_prompt,
            example_prompt_replan=planner_example_prompt_replan,
            tools=tools,
            stop=planner_stop,
        )

        self.agent = LLMCompilerAgent(agent_llm)
        self.joinner_prompt = joinner_prompt
        self.joinner_prompt_final = joinner_prompt_final or joinner_prompt
        self.planner_stream = planner_stream
        self.max_replans = max_replans

        # callbacks
        self.benchmark = benchmark
        if benchmark:
            self.planner_callback = AsyncStatsCallbackHandler(stream=planner_stream)
            self.executor_callback = AsyncStatsCallbackHandler(stream=False)
        else:
            self.planner_callback = None
            self.executor_callback = None

    def get_all_stats(self):
        """
        获取所有统计信息的方法。

        Returns:
            Dict: 包含规划器、执行器以及总体统计信息的字典。
        """
        stats = {}
        if self.benchmark:
            stats["planner"] = self.planner_callback.get_stats()
            stats["executor"] = self.executor_callback.get_stats()
            stats["total"] = {
                k: v + stats["executor"].get(k, 0) for k, v in stats["planner"].items()
            }

        return stats

    def reset_all_stats(self):
        """
        重置所有统计信息的方法。
        """
        if self.planner_callback:
            self.planner_callback.reset()
        if self.executor_callback:
            self.executor_callback.reset()

    @property
    def input_keys(self) -> List[str]:
        """
        获取输入键列表的属性。

        Returns:
            List[str]: 包含输入键的列表，这里只有一个输入键"input"。
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """
        获取输出键列表的属性。

        Returns:
            List[str]: 包含输出键的列表，这里只有一个输出键"output"。
        """
        return [self.output_key]

    # TODO(sk): move all join related functions to a separate class

    
    def _parse_joinner_output(self, raw_answer: str) -> Tuple[str, str, bool]:
        """
        更鲁棒的解析：
        - Thought: 允许多行（取第一段 Thought）
        - Action: Finish/Replan(...) 允许括号内容跨行
        - answer 取括号内完整内容（支持换行）
        """
        raw = (raw_answer or "").strip()

        # 1) Thought：取 Thought: 到下一个 Action: 或文本结束
        thought = ""
        m_thought = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", raw, flags=re.DOTALL)
        if m_thought:
            thought = m_thought.group(1).strip()

        # 2) Action：捕获 Finish(...) 或 Replan(...)
        #    允许括号内容跨行：用 DOTALL + 非贪婪
        answer = ""
        is_replan = False
        m_action = re.search(
            r"Action:\s*(Finish|Replan)\(\s*(.*?)\s*\)\s*$",
            raw,
            flags=re.DOTALL,
        )
        if m_action:
            action_type = m_action.group(1)
            answer = m_action.group(2).strip()
            is_replan = (action_type.lower() == "replan") or (JOINNER_REPLAN in raw)
        else:
            # 兜底：有些模型会写 Action: Finish(xxx) 后面还有别的东西
            m_action2 = re.search(
                r"Action:\s*(Finish|Replan)\(\s*(.*?)\s*\)",
                raw,
                flags=re.DOTALL,
            )
            if m_action2:
                action_type = m_action2.group(1)
                answer = m_action2.group(2).strip()
                is_replan = (action_type.lower() == "replan") or (JOINNER_REPLAN in raw)

        return thought, answer, is_replan

    def _generate_context_for_replanner(
        self, tasks: Mapping[int, Task], joinner_thought: str
    ) -> str:
        """Formatted like this:
        ```
        1. action 1
        Observation: xxx
        2. action 2
        Observation: yyy
        ...
        Thought: joinner_thought
        ```
        """
        previous_plan_and_observations = "\n".join(
            [
                task.get_though_action_observation(
                    include_action=True, include_action_idx=True
                )
                for task in tasks.values()
                if not task.is_join
            ]
        )
        joinner_thought = f"Thought: {joinner_thought}"
        context = "\n\n".join([previous_plan_and_observations, joinner_thought])
        return context

    def _format_contexts(self, contexts: Sequence[str]) -> str:
        """contexts is a list of context
        each context is formatted as the description of _generate_context_for_replanner
        """
        formatted_contexts = ""
        for context in contexts:
            formatted_contexts += f"Previous Plan:\n\n{context}\n\n"
        formatted_contexts += "Current Plan:\n\n"
        return formatted_contexts

    async def join(
        self, input_query: str, agent_scratchpad: str, is_final: bool
    ) -> str:
        if is_final:
            joinner_prompt = self.joinner_prompt_final
        else:
            joinner_prompt = self.joinner_prompt
        prompt = (
            f"{joinner_prompt}\n"  # Instructions and examples
            f"Question: {input_query}\n\n"  # User input query
            f"{agent_scratchpad}\n"  # T-A-O
            # "---\n"
        )
        # log("Joining prompt:\n", prompt, block=True)
        response = await self.agent.arun(
            prompt, callbacks=[self.executor_callback] if self.benchmark else None
        )
        # 将response强制转换为str
        raw_answer = cast(str, response)
        log("Question: \n", input_query, block=True)
        self.log_data["Question"] = input_query
        log("Raw Answer: \n", raw_answer, block=True)
        self.log_data["Raw Answer"] = raw_answer
        thought, answer, is_replan = self._parse_joinner_output(raw_answer)
        if is_final:
            # If final, we don't need to replan
            is_replan = False
            log("Saving logs")
            # 将当前实例的日志添加到全局日志管理器
            GlobalLogManager.add_log(self.log_data)
            
            GlobalLogManager.save_logs_to_file()
            # GlobalLogManager.clear_logs()
            # 如果是最终迭代，保存日志到 all_tasks_logs 文件
            # log("Saving logs to all_tasks_logs.")
            # GlobalLogManager.save_all_tasks_logs(self.log_data)
        return thought, answer, is_replan

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        raise NotImplementedError("LLMCompiler is async only.")

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        contexts = []
        joinner_thought = ""
        agent_scratchpad = ""
        for i in range(self.max_replans):
            is_first_iter = i == 0
            is_final_iter = i == self.max_replans - 1

            task_fetching_unit = TaskFetchingUnit()
            if self.planner_stream:
                task_queue = asyncio.Queue()
                asyncio.create_task(
                    self.planner.aplan(
                        inputs=inputs,
                        task_queue=task_queue,
                        is_replan=not is_first_iter,
                        callbacks=(
                            [self.planner_callback] if self.planner_callback else None
                        ),
                    )
                )
                await task_fetching_unit.aschedule(
                    task_queue=task_queue, func=lambda x: None
                )
            else:
                tasks = await self.planner.plan(
                    inputs=inputs,
                    is_replan=not is_first_iter,
                    # callbacks=run_manager.get_child() if run_manager else None,
                    callbacks=(
                        [self.planner_callback] if self.planner_callback else None
                    ),
                )
                log("Graph of tasks: ", tasks, block=True)
                if self.benchmark:
                    self.planner_callback.additional_fields["num_tasks"] = len(tasks)
                task_fetching_unit.set_tasks(tasks)
                await task_fetching_unit.schedule()
            tasks = task_fetching_unit.tasks

            # collect thought-action-observation
            agent_scratchpad += "\n\n"
            agent_scratchpad += "".join(
                [
                    task.get_though_action_observation(
                        include_action=True, include_thought=True
                    )
                    for task in tasks.values()
                    if not task.is_join
                ]
            )
            agent_scratchpad = agent_scratchpad.strip()

            log("Agent scratchpad:\n", agent_scratchpad, block=True)
            self.log_data["Agent scratchpad"] = agent_scratchpad
            joinner_thought, answer, is_replan = await self.join(
                inputs["input"],
                agent_scratchpad=agent_scratchpad,
                is_final=is_final_iter,
            )
            if not is_replan:
                log("Break out of replan loop.")
                break

            # Collect contexts for the subsequent replanner
            context = self._generate_context_for_replanner(
                tasks=tasks, joinner_thought=joinner_thought
            )
            contexts.append(context)
            formatted_contexts = self._format_contexts(contexts)
            log("Contexts:\n", formatted_contexts, block=True)
            inputs["context"] = formatted_contexts

        if is_final_iter:
            log("Reached max replan limit.")

        return {self.output_key: answer}
