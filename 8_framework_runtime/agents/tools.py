from typing import List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool

from src.tools.base import Tool, tool

# 定义InvalidTool类，它继承自BaseTool，用于在代理遇到无效工具名称时进行处理
class InvalidTool(BaseTool):
    """Tool that is run when invalid tool name is encountered by agent."""

    name: str = "invalid_tool"
    description: str = "Called when tool name is invalid. Suggests valid tool names."

    def _run(
        self,
        requested_tool_name: str,
        available_tool_names: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        同步执行工具的方法。

        当代理尝试使用一个无效的工具名称时，此方法会被调用，用于告知用户所请求的工具名称无效，并提供可用的工具名称列表。

        Args:
            requested_tool_name (str): 代理请求使用的无效工具名称。
            available_tool_names (List[str]): 所有可用的工具名称列表。
            run_manager: 可选的回调管理器，用于在工具运行过程中进行一些额外的操作，如记录日志等。

        Returns:
            str: 包含错误提示信息和可用工具名称建议的字符串。
        """
        # 将可用工具名称列表转换为逗号分隔的字符串形式
        """Use the tool."""
        available_tool_names_str = ", ".join([tool for tool in available_tool_names])
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )

    async def _arun(
        self,
        requested_tool_name: str,
        available_tool_names: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        异步执行工具的方法。

        与同步方法 _run 类似，当代理尝试使用一个无效的工具名称时，此异步方法会被调用，用于告知用户所请求的工具名称无效，并提供可用的工具名称列表。

        Args:
            requested_tool_name (str): 代理请求使用的无效工具名称。
            available_tool_names (List[str]): 所有可用的工具名称列表。
            run_manager: 可选的回调管理器，用于在工具运行过程中进行一些额外的操作，如记录日志等。

        Returns:
            str: 包含错误提示信息和可用工具名称建议的字符串。
        """
        """Use the tool asynchronously."""
        available_tool_names_str = ", ".join([tool for tool in available_tool_names])
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )


__all__ = ["InvalidTool", "BaseTool", "tool", "Tool"]
