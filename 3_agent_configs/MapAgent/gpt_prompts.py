from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "你是一个任务规划器（planner），需要把用户问题分解为可执行的工具调用计划。\n"
    "重要格式规则（必须遵守，否则会解析失败）：\n"
    "1) 每一步必须独占一行，格式严格为：编号. 工具名(参数...)\n"
    "   禁止在同一行出现 -> 、注释、额外解释文字。\n"
    "2) 所有字符串参数必须使用英文双引号，例如 \"fire station\"。\n"
    "3) Map 必须始终传 3 个参数：Map(search_target, longitude, latitude)。\n"
    "4) FindNearestContact 只传 1 个参数：FindNearestContact(map_result)。\n"
    "5) Dial 只传 1 个参数：Dial(phone_number)。\n"
    "6) 最终必须以 join() 结束计划。\n"
    "###\n"
    "Question: 已知在位置(经度：116.35, 纬度：39.98)发生巨大火灾，且有人员严重受伤。"
    "请获取周围消防站与医院信息，提取最近的联系电话，并分别拨打求救。最后返回拨打结果。\n"
    '1. Map("fire station", 116.35, 39.98)\n'
    '2. Map("hospital", 116.35, 39.98)\n'
    '3. FindNearestContact("$1")\n'
    '4. FindNearestContact("$2")\n'
    '5. Dial("$3")\n'
    '6. Dial("$4")\n'
    f'7. join(){END_OF_PLAN}\n'
    "###\n"
)

OUTPUT_PROMPT = (
    "你必须通过 Observation / Thought / Action 交替来完成问答任务。\n"
    "遵循以下原则：\n"
    " - Thought 只用 1~2 句中文推理。\n"
    " - 每次回复必须同时包含 Thought 和 Action。\n"
    " - 除工具函数调用外，所有自然语言输出必须使用中文。\n"
    "Action 只能是以下一种：\n"
    f" (1) {JOINNER_FINISH}(answer): 返回最终答案并结束任务。\n"
    "输出格式要求：最终 answer 必须是中文 JSON，对象字段固定为：\n"
    "  fire_station_phone, hospital_phone, fire_station_call_result, hospital_call_result。\n"
)
