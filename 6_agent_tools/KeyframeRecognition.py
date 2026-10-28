import base64
import asyncio
from openai import OpenAI
import re
import src.utils.reward_manager as reward_manager
import ast

class KeyframeRecognition:
    def __init__(self, model_path="/home/yzt/MiniCPM-V/autodl-tmp/OpenBMB/MiniCPM-V-2_6", api_key="token-abc123", api_base="http://10.200.1.35:9999/v1"):
        self.model_path = model_path
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def encode_image(self, image_path: str) -> str:
        """读取并编码图片为 base64 格式"""
        with open(image_path, 'rb') as file:
            image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')
        return image

    def generate_messages(self, instruction: str, image: str) -> list:
        """构建请求消息"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                        },
                    },
                ],
            }
        ]

    def call_openai_api(self, messages: list) -> str:
        """调用 OpenAI API 获取描述"""
        extra_body = {"stop_token_ids": [151645, 151643]}
        chat_response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0,
            extra_body=extra_body
        )
        return chat_response.choices[0].message.content
  

    _IMG_RE = re.compile(r"(/[^\"'\s]+\.(?:jpg|jpeg|png))", re.IGNORECASE)

    def parse_image_paths(self, image_paths_str) -> list[str]:
        """解析图片路径：支持 str / list / 含 Observation 前缀 / 误拼接表达式"""
        if image_paths_str is None:
            return []

        # 1) 如果上游直接传了 list
        if isinstance(image_paths_str, list):
            return [str(x).strip() for x in image_paths_str if str(x).strip()]

        s = str(image_paths_str).strip()
        if not s:
            return []

        # 2) 去掉常见前缀
        s = s.replace("Observation:", "").strip()

        # 3) 尝试 literal_eval 解析 ["a","b"] / ['a','b']
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [str(x).strip() for x in obj if str(x).strip()]
            if isinstance(obj, str) and obj.strip():
                return [obj.strip()]
        except Exception:
            pass

        # 4) 兜底：用正则从字符串里“抢救”所有图片路径（能救你那种 '[]' + 拼接式）
        hits = self._IMG_RE.findall(s)

        if hits:
            # 去重保序
            seen, out = set(), []
            for p in hits:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out

        # 5) 最后兜底：当作单路径
        return [s]

    def keyframe_recognition_sync(self, image_paths_str: str, instruction: str) -> list:
        image_paths = self.parse_image_paths(image_paths_str)
        results = []

        for image_path in image_paths:
            try:
                # 推荐 URL 方式：image_url = self.image_to_url(image_path)
                # messages = self.generate_messages(instruction, image_url)

                image = self.encode_image(image_path)  # 如果你还用 base64（建议改 URL）
                messages = self.generate_messages(instruction, image)

                result = self.call_openai_api(messages)
                results.append({"image_path": image_path, "result": result})
            except Exception as e:
                # ✅ 错误缩短，别把超长 traceback 传回去
                msg = str(e)
                msg = msg[:300] + ("...(truncated)" if len(msg) > 300 else "")
                results.append({"image_path": image_path, "result": f"Error: {msg}"})

        reward_manager.add_reward(1)
        return results

    async def keyframe_recognition_async(self, image_paths_str: str, instruction: str) -> list:
        """异步调用，支持单个或多个图片路径"""
        image_paths = self.parse_image_paths(image_paths_str)
        loop = asyncio.get_event_loop()
        tasks = []

        async def process_image(image_path):
            try:
                return await loop.run_in_executor(None, self.keyframe_recognition_sync, image_path, instruction)
            except Exception as e:
                return [{"image_path": "Error", "result": f"Error: {str(e)}"}]

        for image_path in image_paths:
            tasks.append(process_image(image_path))

        all_results = await asyncio.gather(*tasks)
        # Flatten the results if necessary
        return [item for sublist in all_results for item in sublist]





# 示例使用
# if __name__ == "__main__":
#     image_path = "['/home/yzt/LLMCompiler/keyframes/car3_frame510_keyframe.jpg','/home/yzt/LLMCompiler/keyframes/car3_frame540_keyframe.jpg']"
#     instruction = "Identify the anomaly subject in the keyframe and provide detailed information such as color, model, etc."
    
#     recognition = KeyframeRecognition()

#     # 同步调用
#     print(recognition.keyframe_recognition_sync(image_path, instruction))

#     # 异步调用
#     result = asyncio.run(recognition.keyframe_recognition_async(image_path, instruction))
#     print(result)
