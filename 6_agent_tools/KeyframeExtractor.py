import os
import torch
import cv2
import numpy as np
import asyncio
from typing import Union
import src.utils.reward_manager as reward_manager

MAX_GPU_DEVICE = "cuda:7"  # 设置 GPU 设备

class KeyframeExtractor:
    def __init__(self, device: str = "cuda:5", save_dir: str = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = "/home/yzt/LLMCompiler/keyframes"
        print(f"使用的设备: {self.device}")

    def _convert_second(self, second: Union[int, str]) -> int:
        """将秒数参数转换为整数"""
        if isinstance(second, int):
            return second
        try:
            return int(second.strip())
        except ValueError:
            raise ValueError(f"无效的秒数参数: {second}")

    def extract_most_useful_keyframe(self, video_path: str):
        """提取视频中的最有用的关键帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                return None

            ret, prev_frame = cap.read()
            if not ret:
                print(f"无法读取首帧: {video_path}")
                cap.release()
                return None

            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_frame_tensor = torch.tensor(prev_frame_gray, dtype=torch.float32).to(self.device)

            max_diff_score = 0
            most_useful_frame = None
            frame_number = 0  # 初始化帧编号

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"完成视频处理: {video_path}")
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_tensor = torch.tensor(frame_gray, dtype=torch.float32).to(self.device)

                diff = torch.abs(frame_tensor - prev_frame_tensor)
                diff_score = torch.mean(diff).item()

                if diff_score > max_diff_score:
                    max_diff_score = diff_score
                    most_useful_frame = frame
                    most_useful_frame_number = frame_number  # 更新最有用帧的编号

                prev_frame_tensor = frame_tensor.clone()
                frame_number += 1  # 更新当前帧编号
            cap.release()
            return most_useful_frame, most_useful_frame_number  # 返回帧和帧编号
        except Exception as e:
            print(f"处理视频时出错: {e}")
            return None, None

    def extract_appointed_keyframe(self, video_path: str, second: int):
        """提取指定秒数的关键帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                return None, None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            cap.release()
            return frame, frame_number  # 返回帧和帧编号
        except Exception as e:
            print(f"处理视频时出错: {e}")
            return None, None

    def save_keyframe(self, video_path: str, frame: np.ndarray, frame_number: int) -> str:
        """保存关键帧为图像文件"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_filename = os.path.join(self.save_dir, f"{video_name}_frame{frame_number}_keyframe.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"保存灰度关键帧: {frame_filename}")
        return f"{frame_filename}"

    def tool_keyframe(self, video_path: str, second: Union[int, str]) -> Union[str, None]:
        """提取视频关键帧并保存"""
        try:
            second = self._convert_second(second)  # 转换为整数
        except ValueError as e:
            print(e)
            return "秒数参数无效，无法处理。"

        if second == -1:
            frame, frame_number = self.extract_most_useful_keyframe(video_path)
        else:
            frame, frame_number = self.extract_appointed_keyframe(video_path, second)

        if frame is not None:
            reward_manager.add_reward(0)
            return self.save_keyframe(video_path, frame, frame_number)
        else:
            print("没有找到有用的帧或处理过程中出现错误。")
            return "没有找到有用的帧或处理过程中出现错误。"

    async def a_tool_keyframe(self, video_path: str, second: Union[int, str]) -> Union[str, None]:
        """异步提取视频关键帧并保存"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.tool_keyframe, video_path, second)



# 示例使用
# if __name__ == "__main__":
#     video_path = "/home/yzt/MiniCPM-V/agent_rt01/Video/car3.mp4"  # 视频路径示例
#     extractor = KeyframeExtractor()

#     # 同步调用
#     print(extractor.tool_keyframe(video_path, 20))  # 提取最有用的关键帧

    # # 异步调用示例
    # result = asyncio.run(extractor.a_tool_keyframe(video_path, 17))  # 提取指定秒数的关键帧
    # print(result)
