import json
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from object_extract import keyframe_recognition
from extract_keyframe import tool_keyframe
# from descrip_picture import describe_image
# from extract_keyframe import extract_most_useful_keyframe
# from video_understand import video_understanding

"""
    1、写文件
    2、读文件
    3、追加的方式写
    4、专业领域知识的获取(网络搜索)
"""

# 定义文件写在哪里
def _get_workdir_root():
    workdir_root = os.environ.get('WORKDIR_ROOT', "./data/llm_result")
    return workdir_root

# 结果存在目录
WORKDIR_ROOT = _get_workdir_root()

# 读文件
def read_file(filename):
    filename = os.path.join(WORKDIR_ROOT, filename)
    if not os.path.exists(filename):
        return f"{filename} not exit, please check file exist before read"
    with open(filename, 'r', encoding="utf-8") as f:
        return "\n".join(f.readlines())

# 追加文件
def append_to_file(filename, content):
    filename = os.path.join(WORKDIR_ROOT, filename)
    if not os.path.exists(filename):
        f"{filename} not exit, please check file exist before read"
    with open(filename, 'a') as f:
        f.write(content)
    return "[SUCCESS]"

# 写文件
def write_to_file(filename, content):
    filename = os.path.join(WORKDIR_ROOT, filename)
    if not os.path.exists(WORKDIR_ROOT):
        os.makedirs(WORKDIR_ROOT)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return "[SUCCESS]"


# 搜索 ：直接使用Langchain里面自带的包
def search(query):
    """
    :param query:
    :return:
    """
    daily = TavilySearchResults(max_results=5)
    try:
        ret = daily.invoke(input=query)
        # print("搜索结果:{}".format(ret))
        # print("\n")
        content_list = []
        """
        # 从哪个网站上获取的内容
        ret = [
            {
                "content": "",
                "url": ""
            }
        ]
        """
        for obj in ret:
            content_list.append(obj["content"])
        # print("搜索结果:{}".format("\n".join(content_list)))
        result = "\n".join(content_list)
        return f"[SUCCESS],result:{result}"
    except Exception as e:
        return "search error:{}".format(e)

# map
def map(search_target, longitude, latitude):
    # 创建一个字典，键是消防站名称，值是[距离(km), 时间(min), 电话]
    if (search_target == "fire station"):
        fire_station = {
            "海淀区公安消防支队首体南路中队": [4.4, 17, "010-68790499"],
            "北京市消防救援总队": [2.7, 14, "010-66167584"],
            "西城区消防救援支队": [2.7, 10, "010-84022119"],
            "亚运村消防救援站": [4.8, 14, "010-64914299"],
            "太阳宫消防救援站": [8.1, 22, "010-64497821"],
            "海淀区消防救援支队": [10.3, 26, "010-82785005"]
        }

        # 输出格式化信息
        result = f"您所处位置（经度：{longitude}, 纬度：{latitude}）附近有以下消防站：\n"
        for service, (distance, time, phone) in fire_station.items():
            result += f"{service}：距离您{distance}km，消防车到达需要至少{time}分钟，消防站电话：{phone}。\n"
        print(result+"\n")
        
        return f"[SUCCESS],已成功获取车祸地点附近的消防站信息:\n{result}"
    
    elif (search_target == "police station"):
        police_station = {
            "北京市公安局海淀分局万寿路派出所": [3.2, 10, "010-68212830"],
            "北京市公安局海淀分局北太平庄派出所": [4.5, 12, "010-62255110"],
            "北京市公安局海淀分局中关村派出所": [5.8, 15, "010-62554600"],
            "北京市公安局海淀分局学院路派出所": [6.1, 13, "010-62317177"],
            "北京市公安局海淀分局清河派出所": [7.3, 18, "010-62913350"],
            "北京市公安局海淀分局上地派出所": [8.4, 16, "010-62962554"]
        }

        # 输出格式化信息
        result = f"您所处位置（经度：{longitude}, 纬度：{latitude}）附近有以下警察局：\n"
        for service, (distance, time, phone) in police_station.items():
            result += f"{service}：距离您{distance}km，警员到达需要至少{time}分钟，警察局电话：{phone}。\n"
        print(result+"\n")
        
        return f"[SUCCESS],已成功获取车祸地点附近的警察局信息:\n{result}"
    
    elif (search_target == "hospital"):
        hospital = {
            "北京大学第三医院": [2.5, 15, "010-82266699"],
            "中国人民解放军总医院第六医学中心": [3.8, 20, "010-66958114"],
            "北京世纪坛医院": [4.2, 18, "010-63925588"],
            "北京肿瘤医院": [5.1, 22, "010-88121122"],
            "清华大学附属北京清华长庚医院": [6.3, 25, "010-56118899"],
            "北京老年医院": [7.5, 30, "010-62402950"]
        }

        # 输出格式化信息
        result = f"您所处位置（经度：{longitude}, 纬度：{latitude}）附近有以下医院：\n"
        for service, (distance, time, phone) in hospital.items():
            result += f"{service}：距离您{distance}km，救护车到达需要至少{time}分钟，医院电话：{phone}。\n"
        print(result+"\n")
        
        return f"[SUCCESS],已成功获取车祸地点附近的医院信息:\n{result}"
    
    else:
        return "[Failure],无法获得该目标地的相关信息，请尝试其他目标地。"

# 打电话
def dial(phone_number):
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
        print("无效的电话号码，请确认后重试。\n")
        return "error,无效的电话号码，请确认后重试。"
    else:
        print(f"已成功拨打电话，联系{service}，请耐心等待救援。\n")
        return f"[SUCCESS],已成功拨打电话，联系{service}，请耐心等待救援。"







# 视频理解
# def video_understand(video_path,instruction):
#     question = instruction

#     # 调用视频理解功能
#     description = video_understanding.get_video_answer(video_path, question)

#     print(description)
#     return description



tools_info = [
    # {
    #     "name": "read_file",
    #     "description": "read file form agent generate, should write file before read",
    #     "args": [
    #         {
    #             "name": "filename",
    #             "type": "string",
    #             "description": "read file name"
    #         }
    #     ]
    # },
    # {
    #     "name": "append_to_file",
    #     "description": "append llm content to file, should write file before read",
    #     "args": [
    #         {
    #             "name": "filename",
    #             "type": "string",
    #             "description": "file name"
    #         },
    #         {
    #             "name": "content",
    #             "type": "string",
    #             "description": "append to file content"
    #         }
    #     ]
    # },
    {
        "name": "write_to_file",
        "description": "write the report to file",
        "args": [
            {
                "name": "filename",
                "type": "string",
                "description": "file name"
            },
            {
                "name": "content",
                "type": "string",
                "description": "write to file content"
            }
        ]
    },
    {
        "name": "finish",
        "description": "End the task and return the result to user",
        "args": [
            {
                "name": "answer",
                "type": "string",
                "description": "the result the user wants"
            }
        ]
    },
    # {
    #     "name": "search",
    #     "description": "this is a search engine, you can gain additional knowledge though this search engine "
    #                    "when you are unsure of large model return",
    #     "args": [
    #         {
    #             "name": "query",
    #             "type": "string",
    #             "description": "search query to look up"
    #         }
    #     ]
    # },
    {
        "name": "map",
        "description": "gain information(distance,name,phone number...) about its surrounding fire stations or hospitals or police stations by search_target, latitude and longitude",
        "args": [
            {
                "name": "search_target",
                "type": "string",
                "description": "search target(fire station or hospital or police station)"
            },
            {
                "name": "longitude",
                "type": "double",
                "description": "longitude"
            },
            {
                "name": "latitude",
                "type": "double",
                "description": "latitude"
            }
        ]
    },
    {
        "name": "dial",
        "description": "call the fire station or hospital or police station for help",
        "args": [
            {
                "name": "phone_number",
                "type": "string",
                "description": "phone number"
            }
        ]
    },
    {
        "name": "keyframe_recognition",
        "description": "you can use this tool to understand and analyze the the key frame of the video",
        "args": [
            {
                "name": "image_path",
                "type": "string",
                "description": "the path of the image"
            },
            {
                "name": "instruction",
                "type": "string",
                "description": "the instruction to llm for containing the information you need"
            }
        ]
    },
    {
        "name": "tool_keyframe",
        "description": "extract the frame of the given second from a video, when second = -1, which will extract the most important frame in the video",
        "args": [
            {
                "name": "video_path",
                "type": "string",
                "description": "the path of the video"
            },
            {
                "name": "second",
                "type": "int",
                "description": "the specified number of second"
            }
        ]
    }
    # {
    #     "name": "video_understand",
    #     "description": "you can use this tool to understand and analyze the video",
    #     "args": [
    #         {
    #             "name": "video_path",
    #             "type": "string",
    #             "description": "the path of the video"
    #         }
    #     ]
    # }
]

tools_map = {
    "write_to_file": write_to_file,
    # "search": search,
    "map": map,
    "dial": dial,
    "keyframe_recognition": keyframe_recognition,
    "tool_keyframe": tool_keyframe
    # "describe_image": describe_image,
    # "extract_most_useful_keyframe": extract_most_useful_keyframe,
    # "video_understand": video_understand
}


def gen_tools_desc():
    """
    生成工具描述
    :return:
    """
    tools_desc = []
    for idx, t in enumerate(tools_info):
        args_desc = []
        for info in t["args"]:
            args_desc.append({
                "name": info["name"],
                "description": info["description"],
                "type": info["type"]
            })
        args_desc = json.dumps(args_desc, ensure_ascii=False)
        tool_desc = f"{idx+1}.{t['name']}:{t['description']}, args: {args_desc}"
        tools_desc.append(tool_desc)
    tools_prompt = "\n".join(tools_desc)
    return tools_prompt



