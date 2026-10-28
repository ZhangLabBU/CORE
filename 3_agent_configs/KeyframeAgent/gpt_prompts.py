from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "Question: 已知存在一个异常视频，路径为 /home/yzt/LLMCompiler/datasets/video_test/car1.mp4，时长5秒。"
    "请提取关键帧并识别每张关键帧中的：车牌号、车辆颜色/车型/受损或起火烟雾、人员动作与受伤迹象。"
    "最终用中文 JSON 数组输出，每个元素包含 image_path、plate、vehicle、people、other_anomaly（无法识别填空字符串）。\n"
    "1. KeyframeExtractor(\"/home/yzt/LLMCompiler/datasets/video_test/car1.mp4\", -1)\n"
    "2. KeyframeExtractor(\"/home/yzt/LLMCompiler/datasets/video_test/car1.mp4\", 2)\n"
    "3. KeyframeExtractor(\"/home/yzt/LLMCompiler/datasets/video_test/car1.mp4\", 4)\n"
    "4. KeyframeRecognition(\"&1\", \"请识别车牌号、车辆颜色/车型/受损或起火烟雾、人员动作与受伤迹象；输出中文 JSON 对象，字段 plate/vehicle/people/other_anomaly。\")\n"
    "5. KeyframeRecognition(\"&2\", \"同上，输出中文 JSON 对象。\")\n"
    "6. KeyframeRecognition(\"&3\", \"同上，输出中文 JSON 对象。\")\n"
    "Thought: 我已获得每张关键帧识别结果，可以汇总为一个中文JSON数组。\n"
    f"7. join(){END_OF_PLAN}\n"
    "###\n"
)


OUTPUT_PROMPT = (
    "你必须通过 Observation / Thought / Action 交替完成任务。\n"
    "约束：除工具调用行外，所有自然语言必须为中文，不得输出任何英文。\n"
    "规则：\n"
    " - 你会收到 Question，以及工具返回的 Observation。\n"
    " - Thought 只写 1~2 句中文推理。\n"
    " - 每次回复必须同时包含 Thought 和 Action。\n"
    " - 如果 Observation 不清晰/不相关，也必须做出最合理推断；Thought 中禁止说“不知道”。\n"
    "Action 只能是以下一种：\n"
    f" (1) {JOINNER_FINISH}(answer)：返回最终答案并结束任务。\n"
    "输出格式：最终 answer 必须是中文 JSON 数组字符串。\n"
    "工具调用格式：函数名(参数1, 参数2, ...)，不要写参数名。\n"
)
