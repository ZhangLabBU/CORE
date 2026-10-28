from typing import Union
import ast
import time
import asyncio
import re
import src.utils.reward_manager as reward_manager

class MapAgent:
    def __init__(self, benchmark=False):
        self.benchmark = benchmark

    def map(self, search_target, longitude, latitude):
        """根据目标地点（如消防站、警察局、医院）返回相关信息。"""

        if search_target == "fire station":
            fire_station = {
                "海淀区公安消防支队首体南路中队": [4.4, 17, "010-68790499"],
                "北京市消防救援总队": [2.7, 14, "010-66167584"],
                "西城区消防救援支队": [2.7, 10, "010-84022119"],
                "亚运村消防救援站": [4.8, 14, "010-64914299"],
                "太阳宫消防救援站": [8.1, 22, "010-64497821"],
                "海淀区消防救援支队": [10.3, 26, "010-82785005"]
            }

            result = f"您所处位置（经度：{longitude}, 纬度：{latitude}）附近有以下消防站：\n"
            for service, (distance, time, phone) in fire_station.items():
                result += f"{service}：距离您{distance}km，消防车到达需要至少{time}分钟，消防站电话：{phone}。\n"

        elif search_target == "police station":
            police_station = {
                "北京市公安局海淀分局万寿路派出所": [3.2, 10, "010-68212830"],
                "北京市公安局海淀分局北太平庄派出所": [4.5, 12, "010-62255110"],
                "北京市公安局海淀分局中关村派出所": [5.8, 15, "010-62554600"],
                "北京市公安局海淀分局学院路派出所": [6.1, 13, "010-62317177"],
                "北京市公安局海淀分局清河派出所": [7.3, 18, "010-62913350"],
                "北京市公安局海淀分局上地派出所": [8.4, 16, "010-62962554"]
            }

            result = f"您所处位置（经度：{longitude}, 纬度：{latitude}）附近有以下警察局：\n"
            for service, (distance, time, phone) in police_station.items():
                result += f"{service}：距离您{distance}km，警员到达需要至少{time}分钟，警察局电话：{phone}。\n"

        elif search_target == "hospital":
            hospital = {
                "北京大学第三医院": [2.5, 15, "010-82266699"],
                "中国人民解放军总医院第六医学中心": [3.8, 20, "010-66958114"],
                "北京世纪坛医院": [4.2, 18, "010-63925588"],
                "北京肿瘤医院": [5.1, 22, "010-88121122"],
                "清华大学附属北京清华长庚医院": [6.3, 25, "010-56118899"],
                "北京老年医院": [7.5, 30, "010-62402950"]
            }

            result = f"您所处位置（经度：{longitude}, 纬度：{latitude}）附近有以下医院：\n"
            for service, (distance, time, phone) in hospital.items():
                result += f"{service}：距离您{distance}km，救护车到达需要至少{time}分钟，医院电话：{phone}。\n"

        else:
            return "[Failure],无法获得该目标地的相关信息，请尝试其他目标地。"

        reward_manager.add_reward(2)
        return f"已成功获取相关信息:\n{result}"

    def dial(self, phone_number):
        """根据电话号码拨打相应服务电话并返回结果。"""
        services = {
            "010-68790499": "海淀区公安消防消防支队首体南路中队",
            "010-66167584": "北京市消防救援总队",
            "010-84022119": "西城区消防救援支队",
            "010-64914299": "亚运村消防救援站",
            "010-64497821": "太阳宫消防救援站",
            "010-82785005": "海淀区消防救援支队",
            "010-68212830": "北京市公安局海淀分局万寿路派出所",
            "010-62255110": "北京市公安局海淀分局北太平庄派出所",
            "010-62554600": "北京市公安局海淀分局中关村派出所",
            "010-62317177": "北京市公安局海淀分局学院路派出所",
            "010-62913350": "北京市公安局海淀分局清河派出所",
            "010-62962554": "北京市公安局海淀分局上地派出所",
            "010-82266699": "北京大学第三医院",
            "010-66958114": "中国人民解放军总医院第六医学中心",
            "010-63925588": "北京世纪坛医院",
            "010-88121122": "北京肿瘤医院",
            "010-56118899": "清华大学附属北京清华长庚医院",
            "010-62402950": "北京老年医院",
        }
        service = services.get(phone_number, "无效的号码")
        if service == "无效的号码":
            return "error,无效的电话号码，请确认后重试。"
        else:
            reward_manager.add_reward(3)
            return f"已成功拨打电话，联系{service}，请耐心等待救援。"

    async def map_async(self, search_target, longitude, latitude):
        """异步调用map方法"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.map, search_target, longitude, latitude)

    async def dial_async(self, phone_number):
        """异步拨打电话"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.dial, phone_number)
    
    def find_fastest_contact(self, map_result: str) -> Union[str, None]:
        """
        从Map返回的结果中提取最快到达的服务的电话号码。
        Args:
            map_result (str): 从map返回的结果。
        Returns:
            str: 最快到达服务的电话号码或错误消息。
        """
        try:
            # 使用正则表达式解析服务名称、到达时间和电话号码
            matches = re.findall(r"(.*?)：.*?需要至少([\d]+)分钟.*?电话：([\d-]+)。", map_result)
            if not matches:
                return "[ERROR],未找到任何电话号码，请检查输入内容。"

            # 按到达时间升序排序，取最快的
            fastest = sorted(matches, key=lambda x: int(x[1]))[0]
            return fastest[2]  # 返回最快到达服务的电话号码
        except Exception as e:
            return f"[ERROR],解析电话号码时出错: {str(e)}"

    async def find_fastest_contact_async(self, map_result: str) -> Union[str, None]:
        """
        异步从Map返回的结果中提取最快到达服务的电话号码。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.find_fastest_contact, map_result)

# if __name__ == "__main__":
#     agent = MapAgent()

#     # 同步调用
#     print(agent.map("fire station", 39.98, 116.35))
#     print(agent.dial("010-68790499"))

#     # 异步调用
#     async def async_example():
#         result = await agent.map_async("hospital", 39.98, 116.35)
#         print(result)
#         result = await agent.dial_async("010-82266699")
#         print(result)

#     asyncio.run(async_example())


# # 示例调用
# if __name__ == "__main__":
#     agent = MapAgent()

#     # 示例返回结果
#     sample_result = (
#         "[SUCCESS],已成功获取相关信息:\n"
#         "您所处位置（经度：39.98, 纬度：116.35）附近有以下医院：\n"
#         "北京大学第三医院：距离您2.5km，救护车到达需要至少15分钟，医院电话：010-82266699。\n"
#         "中国人民解放军总医院第六医学中心：距离您3.8km，救护车到达需要至少20分钟，医院电话：010-66958114。\n"
#         "北京世纪坛医院：距离您4.2km，救护车到达需要至少18分钟，医院电话：010-63925588。\n"
#     )

#     # 同步调用
#     nearest_phone = agent.find_nearest_contact(sample_result)
#     print(f"最近的电话号码是：{nearest_phone}")

#     # 异步调用
#     async def async_example():
#         nearest_phone_async = await agent.find_nearest_contact_async(sample_result)
#         print(f"异步最近的电话号码是：{nearest_phone_async}")

#     asyncio.run(async_example())