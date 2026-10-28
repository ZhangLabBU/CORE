import random
import asyncio
from typing import Union
from datetime import datetime
import re
import src.utils.reward_manager as reward_manager

class MeteorologyAgent:
    def __init__(self):
        self.weather_data = [
            {"condition": "晴", "temperature": 30, "humidity": 40, "wind_speed": 5},
            {"condition": "阴", "temperature": 22, "humidity": 50, "wind_speed": 10},
            {"condition": "小雨", "temperature": 18, "humidity": 80, "wind_speed": 15},
            {"condition": "大雨", "temperature": 16, "humidity": 90, "wind_speed": 20},
            {"condition": "雷暴", "temperature": 25, "humidity": 85, "wind_speed": 30},
            {"condition": "雪", "temperature": -5, "humidity": 70, "wind_speed": 10},
        ]

    @staticmethod
    def get_current_time():
        """
        获取当前时间的字符串表示。
        Returns:
            str: 当前时间，格式为"YYYY-MM-DD HH:MM:SS"。
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def analyze_weather(self, longitude, latitude):
        """
        根据经纬度和当前时间返回随机生成的气象信息。
        Args:
            longitude (float): 经度。
            latitude (float): 纬度。
            time (str): 当前时间。
        Returns:
            str: 气象信息。
        """
        # time = self.get_current_time
        weather = random.choice(self.weather_data)
        return (
            f"当前,您所处位置（经度：{longitude}, 纬度：{latitude}）的气象信息如下：\n"
            f"天气：{weather['condition']}，温度：{weather['temperature']}℃，湿度：{weather['humidity']}%，"
            f"风速：{weather['wind_speed']}km/h。\n"
        )

    def evaluate_impact(self, weather_info, event_type):
        """
        根据气象信息分析对异常情况（火灾/爆炸）的影响。
        Args:
            weather_info (str): 气象信息。
            event_type (str): 异常情况类型（如"火灾"或"爆炸"）。
        Returns:
            str: 对异常情况的分析结果。
        """
        try:
            condition, wind_speed = self._extract_condition_and_wind_speed(weather_info)
            analysis = f"当前天气为{condition}，风速为{wind_speed}km/h。\n"

            if event_type == "fire" :
                if wind_speed > 20:
                    analysis += "风速较高，可能加速火势的蔓延，建议及时疏散并部署风向隔离措施。\n"
                elif wind_speed > 10:
                    analysis += "风速中等，火势可能随风扩散，请重点关注风向。\n"
                else:
                    analysis += "风速较低，火势扩散速度较慢，灭火条件相对有利。\n"

                if condition in ["小雨", "大雨", "雪"]:
                    analysis += "降水有助于减缓火势，请利用降水条件进行灭火。\n"
                elif condition == "雷暴":
                    analysis += "雷暴可能引发新的火源，需重点监控周边环境。\n"
                else:
                    analysis += "天气干燥，火势控制难度较大，请加强灭火措施。\n"

            elif event_type == "explosion":
                if wind_speed > 20:
                    analysis += "风速较高，爆炸后的有害物质可能迅速扩散，请扩大警戒范围。\n"
                elif wind_speed > 10:
                    analysis += "风速中等，有害物质扩散速度适中，请保持监控。\n"
                else:
                    analysis += "风速较低，扩散风险较小，救援条件相对有利。\n"

                if condition in ["小雨", "大雨"]:
                    analysis += "降水可能抑制爆炸后的尘埃扩散，但可能影响地面作业，请注意滑倒风险。\n"
                elif condition == "雷暴":
                    analysis += "雷暴天气可能导致救援设备受损，请确保救援设备防雷。\n"
                else:
                    analysis += "当前气象条件对救援行动无明显阻碍。\n"
            else:
                analysis += "未知的异常情况类型，无法提供针对性分析。\n"

            reward_manager.add_reward(4)
            return analysis
        except Exception as e:
            return f"[ERROR], 分析气象影响时出错: {str(e)}"

    def analyze_rescue_impact(self, weather_info):
        """
        根据气象信息分析对救援工作的影响，并提供建议。
        Args:
            weather_info (str): 气象信息。
        Returns:
            str: 对救援工作的影响和建议。
        """
        try:
            condition, wind_speed = self._extract_condition_and_wind_speed(weather_info)
            advice = f"当前天气为{condition}，风速为{wind_speed}km/h。\n"

            if condition == "晴":
                advice += "天气晴朗，光线充足，适宜救援，但需注意高温对救援人员的体能消耗。\n"
            elif condition == "阴":
                advice += "光线较差，建议配备照明设备，并检查天气变化趋势。\n"
            elif condition == "小雨":
                advice += "小雨天气可能导致地面湿滑，建议穿戴防滑靴并注意防水设备。\n"
            elif condition == "大雨":
                advice += (
                    "大雨可能导致道路积水，建议使用高底盘救援车辆，确保电器设备防水并注意救援人员安全。\n"
                )
            elif condition == "雷暴":
                advice += (
                    "雷暴天气风险较高，应避免高空作业和开阔区域活动，同时检查通信设备的稳定性。\n"
                )
            elif condition == "雪":
                advice += (
                    "雪天可能导致道路湿滑和能见度降低，建议使用雪地专用设备并加强保暖措施。\n"
                )

            if wind_speed > 20:
                advice += "风速较高，可能影响空中救援或加速火势，请考虑调整救援方案。\n"
            elif wind_speed > 10:
                advice += "风速适中，请根据风向安排行动计划，并注意物资固定。\n"
            else:
                advice += "风速较低，救援行动受风速影响较小。\n"

            return advice
        except Exception as e:
            return f"[ERROR], 分析救援影响时出错: {str(e)}"

    def _extract_condition_and_wind_speed(self, weather_info):
        """
        从气象信息中提取天气条件和风速。
        Args:
            weather_info (str): 气象信息。
        Returns:
            tuple: (天气条件, 风速)。
        """
        condition = None
        wind_speed = 0

        if "晴" in weather_info:
            condition = "晴"
        elif "阴" in weather_info:
            condition = "阴"
        elif "小雨" in weather_info:
            condition = "小雨"
        elif "大雨" in weather_info:
            condition = "大雨"
        elif "雷暴" in weather_info:
            condition = "雷暴"
        elif "雪" in weather_info:
            condition = "雪"

        wind_speed_match = re.search(r"风速：(\d+)km/h", weather_info)
        if wind_speed_match:
            wind_speed = int(wind_speed_match.group(1))

        return condition, wind_speed
    
    async def get_current_time_async(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_current_time)


    async def analyze_weather_async(self, longitude, latitude):
        """
        异步调用analyze_weather方法。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_weather, longitude, latitude)

    async def evaluate_impact_async(self, weather_info, event_type):
        """
        异步调用evaluate_impact方法。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate_impact, weather_info, event_type)

    async def analyze_rescue_impact_async(self, weather_info):
        """
        异步调用analyze_rescue_impact方法。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_rescue_impact, weather_info)

    def _extract_condition(self, weather_info):
        """
        从气象信息中提取天气条件。
        """
        if "晴" in weather_info:
            return "晴"
        elif "阴" in weather_info:
            return "阴"
        elif "小雨" in weather_info:
            return "小雨"
        elif "大雨" in weather_info:
            return "大雨"
        elif "雷暴" in weather_info:
            return "雷暴"
        elif "雪" in weather_info:
            return "雪"
        else:
            return "未知"

# # 示例调用
# if __name__ == "__main__":
#     agent = MeteorologyAgent()

#     # 获取当前时间
#     current_time = agent.get_current_time()
#     print(f"当前时间：{current_time}")

#     # 同步调用
#     weather_info = agent.analyze_weather(116.35, 39.98, current_time)
#     print(weather_info)

#     impact_analysis = agent.evaluate_impact(weather_info, "火灾")
#     print(impact_analysis)

#     rescue_impact = agent.analyze_rescue_impact(weather_info)
#     print(rescue_impact)

#     # 异步调用
#     async def async_example():
#         weather_info_async = await agent.analyze_weather_async(116.35, 39.98, current_time)
#         print(weather_info_async)

#         impact_analysis_async = await agent.evaluate_impact_async(weather_info_async, "爆炸")
#         print(impact_analysis_async)

#         rescue_impact_async = await agent.analyze_rescue_impact_async(weather_info_async)
#         print(rescue_impact_async)

#     asyncio.run(async_example())
