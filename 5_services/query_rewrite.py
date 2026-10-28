# services/query_rewrite.py
import dashscope

async def rewrite_query_with_llm(original_query: str, rag_info: dict) -> str:
    """
    使用大模型改写 query，确保最终能够清晰地传递给子 agent
    """
    if not rag_info:
        return original_query

    system_msg = "你是任务指令优化专家，请使用中文优化任务指令。"
    user_msg = f"目标 Agent: {rag_info['agent_name']}\n请改写原始请求：{original_query}"

    resp = dashscope.Generation.call(
        model="qwen-max",
        messages=[{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}],
        result_format='message'
    )

    if resp.status_code == 200:
        return resp.output.choices[0].message.content
    else:
        return original_query
