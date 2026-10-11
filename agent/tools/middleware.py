from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt,load_report_prompt

@wrap_tool_call
def monitor_tool( # 工具监控
        request: ToolCallRequest, #入参
        handler: Callable[[ToolCallRequest], ToolMessage | Command] #执行的函数本身
) -> ToolMessage | Command:
    logger.info(f"[monitor_tool]执行工具：{request.tool_call['name']}")
    logger.info(f"[monitor_tool]传入参数：{request.tool_call['args']}")

    try:
        result= handler(request)
        logger.info(f"[monitor_tool]工具{request.tool_call['name']}调用成功")
        return result
    except Exception as e:
        logger.error(f"[monitor_tool]工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model( #模型监控，在模型执行前执行此函数
        state: AgentState,
        runtime: Runtime   # 整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型，有：{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model] {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt     # 每次生成提示词之前，调用此函数
def report_prompt_switch(request:ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:  #报告生成
        return load_report_prompt()
    return load_system_prompt()


