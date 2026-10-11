from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from agent.intent.classifier import IntentClassifier, Intent
from agent.tools.agent_tools import rag_summarize, get_weather, get_user_location, get_current_month, get_user_id, \
    fetch_external_data
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize, get_weather, get_user_location, get_current_month, get_user_id, \
    fetch_external_data],
            middleware=[monitor_tool,log_before_model,report_prompt_switch]
        )

    def execute_stream(self, query:str):
        # 前置意图识别：识别为报告意图时置 report=True，由 report_prompt_switch 中间件切换提示词
        intent_result = self.intent_classifier.classify(query)

        input_dict={
            "messages":[
                {"role":"user", "content":query}
            ]
        }
        # 参数context就是上下文runtime中的信息，就是做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values",
                                       context={"report": intent_result.intent == Intent.REPORT}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
