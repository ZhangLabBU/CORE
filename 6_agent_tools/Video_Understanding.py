import os
import torch
import time
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from typing import Union
import asyncio
import cv2
import src.utils.reward_manager as reward_manager
# 设置设备为 CUDA 5
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

MAX_NUM_FRAMES = 64
TARGET_WIDTH = 224  # 设定目标宽度
TARGET_HEIGHT = 224  # 设定目标高度

class VideoUnderstanding:
    all_times = []

    def __init__(self, model, tokenizer, benchmark=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.benchmark = benchmark

    def encode_video(self, video_path):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()

        # 将所有帧调整为相同的大小
        frames = [cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT)) for frame in frames]
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        print('num frames:', len(frames))
        return frames
    
    # def encode_video2(self, video_path, frame_number=None, save_frame=False):
    #     def uniform_sample(l, n):
    #         gap = len(l) / n
    #         idxs = [int(i * gap + gap / 2) for i in range(n)]
    #         return [l[i] for i in idxs]

    #     vr = VideoReader(video_path, ctx=cpu(0))
    #     sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    #     frame_idx = [i for i in range(0, len(vr), sample_fps)]
        
    #     if len(frame_idx) > MAX_NUM_FRAMES:
    #         frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        
    #     # 如果指定了某一帧（如第6帧），提取该帧
    #     if frame_number is not None:
    #         frame_idx = [frame_number]  # 只取第frame_number帧
        
    #     frames = vr.get_batch(frame_idx).asnumpy()
        
    #     # 将所有帧调整为相同的大小
    #     frames = [cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT)) for frame in frames]
    #     frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        
    #     # 如果需要保存第6帧为图片
    #     if save_frame and frame_number is not None:
    #         save_path = f"/home/yzt/LLMCompiler/keyframes/frame_{frame_number}.jpg"
    #         frames[0].save(save_path, "JPEG")
    #         print(f"Frame {frame_number} saved as {save_path}")
        
    #     print('num frames:', len(frames))
    #     return frames


    def get_video_answer(self, video_path, question):
        frames = self.encode_video(video_path)
        prompt_question = f"Please describe the content of the video in detail.And answer the question:{question}"
        msgs = [{'role': 'user', 'content': frames + [prompt_question]}]
        params = {"use_image_id": False, "max_slice_nums": 2}

        answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, **params)
        
        return answer

    def analyze_video(self, video_path: str, question: str, is_retry: bool = False) -> Union[str, str]:
        """同步分析视频并回答问题。"""
        s = time.time()
        try:
            answer = self.get_video_answer(video_path, question)
            if self.benchmark and not is_retry:
                self.all_times.append(round(time.time() - s, 2))
            reward_manager.add_reward(6)
            return answer
        except Exception as e:
            return f"视频分析错误: {e}"

    async def a_analyze_video(self, video_path: str, question: str, is_retry: bool = False) -> Union[str, str]:
        """异步分析视频并回答问题。"""
        s = time.time()
        try:
            answer = await self._async_analyze(video_path, question)
            if self.benchmark and not is_retry:
                self.all_times.append(round(time.time() - s, 2))
            reward_manager.add_reward(6)
            return answer
        except Exception as e:
            return f"视频分析错误: {e}"

    async def _async_analyze(self, video_path: str, question: str) -> str:
        """异步辅助分析方法。"""
        loop = asyncio.Future()
        answer = self.get_video_answer(video_path, question)
        loop.set_result(answer)
        return await loop

# # 示例用法：
# # 模型权重文件目录
# MODEL_DIR = '/home/yzt/MiniCPM-V/autodl-tmp/OpenBMB/MiniCPM-V-2_6'

# # 设置设备为 CUDA 5
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# print(f"使用的设备: {device}")

# # 加载模型和分词器
# model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
# model = model.eval().to(device)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
# # 创建 VideoAgent 实例
# video_agent = VideoUnderstanding(model, tokenizer, device, benchmark=True)


# # 同步视频分析
# video_path = "/home/yzt/MiniCPM-V/agent_rt01/Video/car3.mp4"
# # 假设我们想提取第6帧并保存
# # frames = video_agent.encode_video2(video_path, frame_number=10, save_frame=True)
# question = "在第几张图片中车辆发生了侧翻？"
# result = video_agent.analyze_video(video_path, question)
# print(result)

# # 异步视频分析
# result = asyncio.run(video_agent.a_analyze_video(video_path, question))
# print(result)
