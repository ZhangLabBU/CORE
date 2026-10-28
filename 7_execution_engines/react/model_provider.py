import os, json
import dashscope
from prompt import user_prompt
from dashscope.api_entities.dashscope_response import Message
from zhipuai import ZhipuAI


class ModelProvider(object):
    def __init__(self):
        self.api_key = os.environ.get('Zhi_Pu_API_KEY')
        self.model_name = os.environ.get('MODEL_NAME')
        self.temperature = 0
        self._client = ZhipuAI(api_key=self.api_key).chat.completions
        self.max_retry_time = 3

    def chat(self, prompt, chat_history):
        cur_retry_time = 0
        while cur_retry_time < self.max_retry_time:
            cur_retry_time += 1
            try:
                messages = [
                    Message(role="system", content=prompt)
                ]
                for his in chat_history:
                    messages.append(Message(role="user", content=his[0]))
                    messages.append(Message(role="system", content=his[1]))

                # 最后1条信息是用户的输入
                messages.append(Message(role="user", content=user_prompt))
                response = self._client.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature
                )
                print("response:{}".format(response.choices[0].message.content))

                # 大模型返回格式不标准，需要先处理一下
                clean_response = response.choices[0].message.content.strip()
                clean_response = clean_response.replace('```json', '').replace('```', '').strip()
                content = json.loads(clean_response)
                return content
            except Exception as e:
                print("call llm exception:{}".format(e))
        return {}
