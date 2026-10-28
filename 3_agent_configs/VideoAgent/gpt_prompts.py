from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "你是一个任务规划器（planner），需要把用户问题分解为可执行的工具调用计划。\n"
    "重要格式规则（必须遵守，否则会解析失败）：\n"
    "1) 每一步必须独占一行，严格格式：编号. 工具名(参数...)\n"
    "2) 禁止在工具调用行里出现 ->、注释、额外解释文字。\n"
    "3) 字符串参数必须使用英文双引号。\n"
    "4) KeyframeRecognition 的参数顺序固定为：KeyframeRecognition(image_path, instruction)。\n"
    "5) 最终必须以 join() 结束计划。\n"
    "###\n"

    # ===== 示例 1：总长 5 秒（<=10s：2帧，建议 1/3 和 2/3）=====
    "Question: 已知一个异常视频，视频路径为 \"path/video.mp4\"，视频总长为 5 秒。请问视频出现了什么异常事故？\n"
    '1. VideoUnderstanding("path/video.mp4", "视频出现了什么异常事故？")\n'
    '2. KeyframeExtractor("path/video.mp4", 2)\n'
    '3. KeyframeExtractor("path/video.mp4", 4)\n'
    '4. KeyframeRecognition("$2", "识别该关键帧中的事故类型与场景（如追尾、翻车、起火、碰撞等），用中文简洁描述。")\n'
    '5. KeyframeRecognition("$3", "识别该关键帧中的人物行为与受伤迹象（人数、姿态、倒地、血迹、求助等），用中文简洁描述。")\n'
    f'6. join(){END_OF_PLAN}\n'
    "###\n"

    # ===== 示例 2：总长 15 秒（>10s：3帧，等间隔 5/10/15）=====
    "Question: 已知一个异常视频，视频路径为 \"path/video.mp4\"，视频总长为 15 秒。请问视频出现了什么异常事故？\n"
    '1. VideoUnderstanding("path/video.mp4", "视频出现了什么异常事故？")\n'
    '2. KeyframeExtractor("path/video.mp4", 5)\n'
    '3. KeyframeExtractor("path/video.mp4", 10)\n'
    '4. KeyframeExtractor("path/video.mp4", 15)\n'
    '5. KeyframeRecognition("$2", "分析第 5 秒关键帧：事故类型、车辆状态、道路环境、是否有烟雾/火焰等（中文）。")\n'
    '6. KeyframeRecognition("$3", "分析第 10 秒关键帧：事故进展、是否二次碰撞、是否有人倒地/受伤（中文）。")\n'
    '7. KeyframeRecognition("$4", "分析第 15 秒关键帧：现场处置/人员聚集/救援迹象，给出中文总结。")\n'
    f'8. join(){END_OF_PLAN}\n'
    "###\n"

    # ===== 示例 3：总长 30 秒（车辆类型）=====
    "Question: 已知一个视频，视频路径为 \"path/video.mp4\"，视频总长为 30 秒。请问视频中出现了哪些类型的车辆？\n"
    '1. VideoUnderstanding("path/video.mp4", "视频中出现了哪些类型的车辆？")\n'
    '2. KeyframeExtractor("path/video.mp4", 10)\n'
    '3. KeyframeExtractor("path/video.mp4", 20)\n'
    '4. KeyframeExtractor("path/video.mp4", 30)\n'
    '5. KeyframeRecognition("$2", "识别第 10 秒：车辆类型（轿车/SUV/卡车/公交/摩托等）、颜色、明显车型特征、是否可见车牌（中文）。")\n'
    '6. KeyframeRecognition("$3", "识别第 20 秒：车辆类型、颜色、车牌是否可见（中文）。")\n'
    '7. KeyframeRecognition("$4", "识别第 30 秒：车辆类型、颜色、车牌是否可见（中文）。")\n'
    f'8. join(){END_OF_PLAN}\n'
    "###\n"
)
OUTPUT_PROMPT = (
    "你必须通过 Observation / Thought / Action 交替完成任务。\n"
    "强约束：除工具调用行外，所有自然语言必须为中文，不得输出英文。\n"
    "Thought 仅 1~2 句中文。\n"
    "Action 只能是：\n"
    f" (1) {JOINNER_FINISH}(answer)：返回最终答案并结束任务。\n"
    "输出要求：最终 answer 用中文总结事故类型、关键证据（来自关键帧），并说明是否有人受伤及依据。\n"
)
