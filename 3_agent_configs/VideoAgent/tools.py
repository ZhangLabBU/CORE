from src.agents.tools import Tool
from src.docstore.TavilySearch import TavilySearch
from src.docstore.Video_Understanding import VideoUnderstanding
from src.docstore.KeyframeExtractor import KeyframeExtractor
from src.docstore.KeyframeRecognition import KeyframeRecognition
import time
import time
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from typing import Union
import asyncio
import torch
import json

def _q(s: str) -> str:
    s = "" if s is None else str(s)
    return json.dumps(s, ensure_ascii=False)

def _fmt_arg(x) -> str:
    if x is None:
        return '""'
    if isinstance(x, str):
        t = x.strip()
        if t.startswith("$"):   # $1 $2 这种依赖变量别加引号
            return t
        return _q(t)            # 普通字符串一律转成合法 "..."
    return str(x)

# 模型权重文件目录
MODEL_DIR = '/home/yzt/MiniCPM-V/autodl-tmp/OpenBMB/MiniCPM-V-2_6'

# 设置设备为 CUDA 5
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# print(f"使用的设备: {device}")

# 加载模型和分词器
model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
# model = model.eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)

video_understand = VideoUnderstanding(model, tokenizer,benchmark=True)

# tavily_search = TavilySearch()

extractor = KeyframeExtractor()

recognition = KeyframeRecognition()

tools = [
    Tool(
        name="VideoUnderstanding",
        func=video_understand.a_analyze_video,
        description=(
            "VideoUnderstanding(video_path: str, question: str) -> str:\n"
            " - Analyzes the given video based on the provided question to extract specific insights.\n"
            " - Supports asynchronous analysis of videos, such as answering questions about video content.\n"
            " - `video_path`: Path to the video file to be analyzed, e.g., '/path/to/video.mp4'.\n"
            " - `question`: A specific question about the video's content, e.g., 'What is happening in this video?'.\n"
            " - Returns a dictionary containing the analysis results for the given question."
        ),
        stringify_rule=lambda args: f"VideoUnderstanding({_fmt_arg(args[0])}, {_fmt_arg(args[1])})",
    ),
        Tool(
        name="KeyframeExtractor",
        func=extractor.a_tool_keyframe,
        description=(
            "KeyframeExtractor(video_path: str, second: int) -> str:\n"
            " - Extracts a keyframe from the given video at the specified time or the most useful keyframe.\n"
            " - Supports both synchronous and asynchronous extraction of keyframes.\n"
            " - `video_path`: Path to the video file from which to extract the keyframe, e.g., '/path/to/video.mp4'.\n"
            " - `second`: The specific second in the video to extract the frame. If set to -1, the most useful keyframe is extracted based on frame differences.\n"
            " - Returns a string indicating whether the keyframe extraction was successful and the location where the keyframe is saved."
        ),
        stringify_rule=lambda args: f"KeyframeExtractor({_fmt_arg(args[0])}, {_fmt_arg(args[1])})",
    ),
    Tool(
        name="KeyframeRecognition",
        func=recognition.keyframe_recognition_async,
        description=(
            "KeyframeRecognition(image_path: str, instruction: str) -> str:\n"
            " - Analyzes the given image and provides a detailed description based on the provided instruction.\n"
            " - Supports both synchronous and asynchronous recognition of keyframes.\n"
            " - `image_path`: Path to the image file to be analyzed, e.g., '/path/to/image.jpg'.\n"
            " - `instruction`: A specific instruction describing what to look for in the image, e.g., 'Describe the people in the image'.\n"
            " - Returns a string containing the description of the image based on the instruction."
        ),
        stringify_rule=lambda args: f"KeyframeRecognition({_fmt_arg(args[0])}, {_fmt_arg(args[1])})",
    ),
]
