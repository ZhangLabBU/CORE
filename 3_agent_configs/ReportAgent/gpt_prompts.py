from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "你是一个任务规划器（planner），需要把用户问题分解为可执行的工具调用计划。\n"
    "重要格式规则（必须遵守，否则会解析失败）：\n"
    "1) 每一步必须独占一行，严格格式：编号. 工具名(参数...)\n"
    "2) 字符串参数必须使用英文双引号。\n"
    "3) 禁止使用参数名（如 logs=、report_content=），只允许位置参数。\n"
    "4) 禁止在工具调用行里添加 ->、注释、解释文字。\n"
    "5) 最终必须以 join() 结束。\n"
    "###\n"
    "示例：\n"
    '1. ReadFile("LLMCompiler/result/logs/logs_test.json")\n'
    '2. GenerateReport("$1")\n'
    '3. GetImagePaths("$1")\n'
    '4. WriteReport("$2", $3)\n'
    f"5. join(){END_OF_PLAN}\n"
    "###\n"
)



OUTPUT_PROMPT = (
    "你必须通过 Observation / Thought / Action 交替完成任务。\n"
    "约束：除工具调用行外，所有自然语言必须为中文，不得输出任何英文。\n"
    "规则：\n"
    " - Thought 只写 1~2 句中文推理。\n"
    " - Action 只能是以下一种：\n"
    f"   (1) {JOINNER_FINISH}(answer)：返回最终答案并结束任务。\n"
    " - 最终 answer 需要说明：报告是否生成成功，以及生成的 docx 文件路径。\n"
)
