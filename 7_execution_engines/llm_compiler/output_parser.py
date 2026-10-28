import ast
import re
from typing import Any, Sequence, Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import OutputParserException

from src.llm_compiler.task_fetching_unit import Task
from src.tools.base import StructuredTool, Tool

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"(?:^|\n)\s*(\d+)\.\s*([A-Za-z_]\w*)\((.*?)\)\s*(?:#\w+)?\s*(?=\n|$)"
# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"

END_OF_PLAN = "<END_OF_PLAN>"


def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


class LLMCompilerPlanParser(AgentOutputParser, extra="allow"):
    """Planning output parser."""

    def __init__(self, tools: Sequence[Union[Tool, StructuredTool]], **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def parse(self, text: str) -> dict[int, Task]:
        # ✅ planner 可能输出 1. ... -> 2. ... 这种一行串联，先打散
        text = re.sub(r"\s*(?:->|→)\s*", "\n", text)

        pattern = re.compile(rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}", re.DOTALL)
        matches = pattern.findall(text)

        graph_dict: dict[int, Task] = {}

        for match in matches:
            # match 可能是 (thought, idx, tool_name, args) 或者 (thought, idx, tool_name, args, ...)
            # 我们按你定义的 THOUGHT_PATTERN + ACTION_PATTERN 来解包
            thought = match[0] if len(match) > 0 else ""
            idx = int(match[1])
            tool_name = match[2]
            args = match[3]

            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )

            graph_dict[idx] = task
            if task.is_join:
                break

        return graph_dict

        # 1. search("Ronaldo number of kids") -> 1, "search", '"Ronaldo number of kids"'
        # pattern = r"(\d+)\. (\w+)\(([^)]+)\)"
        text = re.sub(r"\s*(?:->|→)\s*", "\n", text)
        pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
        matches = re.findall(pattern, text)

        graph_dict = {}

        for match in matches:
            # idx = 1, function = "search", args = "Ronaldo number of kids"
            # thought will be the preceding thought, if any, otherwise an empty string
            thought, idx, tool_name, args, _ = match
            idx = int(idx)

            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )

            graph_dict[idx] = task
            if task.is_join:
                break

        return graph_dict


### Helper functions


def _parse_llm_compiler_action_args(args: str) -> list:
    """Parse arguments from a string, removing parameter names."""
    if args == "":
        return []  # Return an empty list for empty input

    # in_quote = False
    # parts = []
    # current_part = ""
    # for char in args:
    #     if char == '"' and not in_quote:
    #         in_quote = True
    #     elif char == '"' and in_quote:
    #         in_quote = False
    #     elif char == ',' and not in_quote:
    #         parts.append(current_part.strip())
    #         current_part = ""
    #     else:
    #         current_part += char
    # parts.append(current_part.strip())  # Add the last part
    
    # Use a regular expression to split by commas outside of quotes and parentheses
    parts = re.split(r',(?=(?:[^"\'()]*["\'](?:\\.|[^"\'])*["\']|[^"\'()]*\([^)]*\))*[^"\'()]*$)', args)

    # Remove parameter names (e.g., "param=value" -> "value")
    cleaned_parts = []
    for part in parts:
        if '=' in part:
            _, value = part.split('=', 1)
            cleaned_parts.append(value.strip())
        else:
            cleaned_parts.append(part.strip())

    # Attempt to evaluate as literal, but handle potential errors gracefully
    try:
        parsed_args = [ast.literal_eval(part) if part.startswith(('"', "'")) else part for part in cleaned_parts]
    except (ValueError, SyntaxError) as e:
        print(f"Literal evaluation error: {e}")
        parsed_args = cleaned_parts

    return parsed_args


def _find_tool(
    tool_name: str, tools: Sequence[Union[Tool, StructuredTool]]
) -> Union[Tool, StructuredTool]:
    """Find a tool by name.

    Args:
        tool_name: Name of the tool to find.

    Returns:
        Tool or StructuredTool.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise OutputParserException(f"Tool {tool_name} not found.")


def _get_dependencies_from_graph(
    idx: int, tool_name: str, args: Sequence[Any]
) -> dict[str, list[str]]:
    """Get dependencies from a graph."""
    if tool_name == "join":
        # depends on the previous step
        dependencies = list(range(1, idx))
    else:
        # define dependencies based on the dependency rule in tool_definitions.py
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, args)]

    return dependencies


def instantiate_task(
    tools: Sequence[Union[Tool, StructuredTool]],
    idx: int,
    tool_name: str,
    args: str,
    thought: str,
) -> Task:
    dependencies = _get_dependencies_from_graph(idx, tool_name, args)
    args = _parse_llm_compiler_action_args(args)
    if tool_name == "join":
        # join does not have a tool
        tool_func = lambda x: None
        stringify_rule = None
    else:
        tool = _find_tool(tool_name, tools)
        tool_func = tool.func
        stringify_rule = tool.stringify_rule
    return Task(
        idx=idx,
        name=tool_name,
        tool=tool_func,
        args=args,
        dependencies=dependencies,
        stringify_rule=stringify_rule,
        thought=thought,
        is_join=tool_name == "join",
    )
