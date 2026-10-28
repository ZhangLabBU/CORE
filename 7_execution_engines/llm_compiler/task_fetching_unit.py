from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Optional, List

from src.utils.logger_utils import log

SCHEDULING_INTERVAL = 0.01  # seconds
STUCK_TIMEOUT_SEC = 60      # ✅ 之前是 5 秒，LLM 很容易超过；这里给 60 秒更合理


def _default_stringify_rule_for_arguments(args):
    if len(args) == 1:
        return str(args[0])
    else:
        return str(tuple(args))


def _replace_arg_mask_with_real_value(args, dependencies: List[int], tasks: Dict[int, "Task"]):
    # list/tuple 递归
    if isinstance(args, (list, tuple)):
        return type(args)(
            _replace_arg_mask_with_real_value(item, dependencies, tasks)
            for item in args
        )

    # str：替换 $1 / ${1}
    if isinstance(args, str):
        for dependency in sorted(dependencies, reverse=True):
            for arg_mask in [f"${{{dependency}}}", f"${dependency}"]:
                if arg_mask in args:
                    dep_task = tasks.get(dependency)
                    if dep_task and dep_task.observation is not None:
                        args = args.replace(arg_mask, str(dep_task.observation))
        return args

    return args


@dataclass
class Task:
    idx: int
    name: str
    tool: Callable
    args: Collection[Any]
    dependencies: Collection[int]
    stringify_rule: Optional[Callable] = None
    thought: Optional[str] = None
    observation: Optional[str] = None
    is_join: bool = False

    async def __call__(self) -> Any:
        log("running task")
        try:
            x = await self.tool(*self.args)
            log("done task")
            return x
        except Exception as e:
            log(f"Task {self.idx} failed with error: {e}")
            self.observation = f"Error: {e}"
            return None

    def get_though_action_observation(
        self, include_action=True, include_thought=True, include_action_idx=False
    ) -> str:
        s = ""
        if self.thought and include_thought:
            s += f"Thought: {self.thought}\n"
        if include_action:
            idx = f"{self.idx}. " if include_action_idx else ""
            if self.stringify_rule:
                s += f"{idx}{self.stringify_rule(self.args)}\n"
            else:
                s += f"{idx}{self.name}{_default_stringify_rule_for_arguments(self.args)}\n"
        if self.observation is not None:
            s += f"Observation: {self.observation}\n"
        return s


class TaskFetchingUnit:
    tasks: Dict[int, Task]
    tasks_done: Dict[int, asyncio.Event]
    remaining_tasks: set[int]

    def __init__(self):
        self.tasks = {}
        self.tasks_done = {}
        self.remaining_tasks = set()

        # ✅ 新增：跟踪正在运行中的任务
        self._in_flight: Dict[int, asyncio.Task] = {}

    def set_tasks(self, tasks: dict[int, Task]):
        self.tasks.update(tasks)
        for task_idx in tasks:
            self.tasks_done.setdefault(task_idx, asyncio.Event())
            self.remaining_tasks.add(task_idx)

    def _all_tasks_done(self) -> bool:
        return all(ev.is_set() for ev in self.tasks_done.values())

    def _sanitize_dependencies(self):
        existing = set(self.tasks.keys())
        for tid, task in self.tasks.items():
            if not task.dependencies:
                continue
            missing = [d for d in task.dependencies if d not in existing]
            if missing:
                log(f"[WARN] task {tid} deps not arrived yet: {missing}")

    def _get_all_executable_tasks(self) -> list[int]:
        self._sanitize_dependencies()

        executable: list[int] = []
        for task_id in list(self.remaining_tasks):
            deps = list(self.tasks[task_id].dependencies or [])
            ok = True
            for d in deps:
                if d not in self.tasks_done:
                    ok = False
                    break
                if not self.tasks_done[d].is_set():
                    ok = False
                    break
            if ok:
                executable.append(task_id)
        return executable

    def _preprocess_args(self, task: Task):
        args = []
        for arg in task.args:
            arg = _replace_arg_mask_with_real_value(arg, list(task.dependencies or []), self.tasks)
            args.append(arg)
        task.args = args

    async def _run_task(self, task: Task):
        try:
            self._preprocess_args(task)
            if not task.is_join:
                observation = await task()
                task.observation = observation
        finally:
            self.tasks_done.setdefault(task.idx, asyncio.Event()).set()
            # ✅ 结束后从 in_flight 移除
            self._in_flight.pop(task.idx, None)

    def _start_task(self, task_id: int):
        t = asyncio.create_task(self._run_task(self.tasks[task_id]))
        self._in_flight[task_id] = t
        self.remaining_tasks.discard(task_id)

    async def schedule(self):
        idle_start = None

        while not self._all_tasks_done():
            executable_tasks = self._get_all_executable_tasks()

            if executable_tasks:
                idle_start = None
                for task_id in executable_tasks:
                    self._start_task(task_id)

            else:
                # ✅ 没有可执行任务：
                # - 如果还有任务在跑：正常等待（不要判 stuck）
                # - 如果没有任务在跑，但还有 remaining_tasks：才可能死锁
                if self._in_flight:
                    await asyncio.sleep(SCHEDULING_INTERVAL)
                    continue

                if not self.remaining_tasks:
                    await asyncio.sleep(SCHEDULING_INTERVAL)
                    continue

                # 真的可能卡住：开始计时
                if idle_start is None:
                    idle_start = asyncio.get_event_loop().time()

                if (asyncio.get_event_loop().time() - idle_start) > STUCK_TIMEOUT_SEC:
                    stuck = list(self.remaining_tasks)
                    log(f"[ERROR] scheduling stuck (>{STUCK_TIMEOUT_SEC}s), remaining={stuck}")
                    break

            await asyncio.sleep(SCHEDULING_INTERVAL)

    async def aschedule(self, task_queue: asyncio.Queue[Optional[Task]], func):
        no_more_tasks = False
        idle_start = None

        while True:
            if not no_more_tasks:
                task = await task_queue.get()
                if task is None:
                    no_more_tasks = True
                else:
                    self.set_tasks({task.idx: task})

            executable_tasks = self._get_all_executable_tasks()

            if executable_tasks:
                idle_start = None
                for task_id in executable_tasks:
                    self._start_task(task_id)

            else:
                if no_more_tasks and self._all_tasks_done():
                    break

                if self._in_flight:
                    await asyncio.sleep(SCHEDULING_INTERVAL)
                    continue

                if no_more_tasks and self.remaining_tasks:
                    if idle_start is None:
                        idle_start = asyncio.get_event_loop().time()
                    if (asyncio.get_event_loop().time() - idle_start) > STUCK_TIMEOUT_SEC:
                        stuck = list(self.remaining_tasks)
                        log(f"[ERROR] aschedule stuck (>{STUCK_TIMEOUT_SEC}s), remaining={stuck}")
                        break

                await asyncio.sleep(SCHEDULING_INTERVAL)
