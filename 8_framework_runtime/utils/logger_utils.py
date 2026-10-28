import json
import time
from collections import defaultdict
from threading import Lock
import os

import numpy as np

# Global variable to toggle logging
LOG_ENABLED = True



class GlobalLogManager:
    """全局日志管理类，负责统一管理日志的写入。"""
    _log_data = []  # 用于存储所有日志
    _file_path = "/home/yzt/AgentDNS/result/logs/logs_test.json"  # 子 Agent 通信日志文件路径
    _all_tasks_log_path = "/home/yzt/AgentDNS/result/logs/all_tasks_logs.json"  # 全部任务日志文件路径
    _lock = Lock()  # 确保多线程环境下的线程安全

    @classmethod
    def add_log(cls, log_entry: dict):
        """添加日志到全局日志列表中。"""
        with cls._lock:
            cls._log_data.append(log_entry)
            print(f"日志添加成功: {log_entry}")  # 调试打印

    @classmethod
    def save_logs_to_file(cls):
        """将当前任务日志写入 log_test 文件。"""
        with cls._lock:
            with open(cls._file_path, "w", encoding="utf-8") as f:
                json.dump(cls._log_data, f, ensure_ascii=False, indent=4)

    @classmethod
    def clear_log_test(cls):
        """清空 log_test 文件内容。"""
        with cls._lock:
            with open(cls._file_path, "w", encoding="utf-8") as f:
                f.write("[]")  # 清空文件并写入空数组
            cls._log_data = []  # 同时清空内存中的日志数据
            print("log_test 文件已清空。")

    @classmethod
    def save_all_tasks_logs(cls, task_logs: dict):
        """将单个任务的日志追加到 all_tasks_logs 文件中。"""
        with cls._lock:
            if not os.path.exists(cls._all_tasks_log_path):
                with open(cls._all_tasks_log_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)

            with open(cls._all_tasks_log_path, "r+", encoding="utf-8") as f:
                try:
                    existing_logs = json.load(f)
                except json.JSONDecodeError:
                    existing_logs = []
                existing_logs.append(task_logs)
                f.seek(0)
                json.dump(existing_logs, f, ensure_ascii=False, indent=4)
                print("任务日志已追加到 all_tasks_logs 文件。")

    @classmethod
    def a_save_logs_to_file(cls):
        """将当前日志续写到 log_test 文件中。"""
        with cls._lock:
            if not os.path.exists(cls._file_path):
                with open(cls._file_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)

            with open(cls._file_path, "r+", encoding="utf-8") as f:
                try:
                    existing_logs = json.load(f)
                except json.JSONDecodeError:
                    existing_logs = []
                existing_logs.extend(cls._log_data)
                f.seek(0)
                json.dump(existing_logs, f, ensure_ascii=False, indent=4)
                print("日志已续写到 log_test 文件。")

    @classmethod
    def clear_logs(cls):
        """清除当前内存中的日志数据。"""
        with cls._lock:
            cls._log_data = []
            print("内存中的日志数据已清空。")

class Logger:
    def __init__(self) -> None:
        self._latency_dict = defaultdict(list)
        self._answer_dict = defaultdict(list)
        self._label_dict = defaultdict(list)
        

    def log(self, latency: float, answer: str, label: str, key: str) -> None:
        self._latency_dict[key].append(latency)
        self._answer_dict[key].append(answer)
        self._label_dict[key].append(label)

    def _get_mean_latency(self, key: str) -> float:
        latency_array = np.array(self._latency_dict[key])
        return latency_array.mean(), latency_array.std()

    def _get_accuracy(self, key: str) -> float:
        answer_array = np.array(self._answer_dict[key])
        label_array = np.array(self._label_dict[key])
        return (answer_array == label_array).mean()

    def get_results(self, key: str) -> dict:
        mean_latency, std_latency = self._get_mean_latency(key)
        accuracy = self._get_accuracy(key)
        return {
            "mean_latency": mean_latency,
            "std_latency": std_latency,
            "accuracy": accuracy,
        }

    def save_result(self, key: str, path: str):
        with open(f"{path}/dev_react_results.csv", "w") as f:
            for i in range(len(self._answer_dict[key])):
                f.write(f"{self._answer_dict[key][i]},{self._latency_dict[key][i]}\n")


def get_logger() -> Logger:
    return Logger()


# Custom print function to toggle logging


def enable_logging(enable=True):
    """Toggle logging on or off based on the given argument."""
    global LOG_ENABLED
    LOG_ENABLED = enable


def log(*args, block=False, **kwargs):
    """Print the given string only if logging is enabled."""
    if LOG_ENABLED:
        if block:
            print("=" * 80)
        print(*args, **kwargs)
        if block:
            print("=" * 80)


def flush_results(save_path, results):
    print("Saving results")
    json.dump(results, open(save_path, "w"), indent=4)
