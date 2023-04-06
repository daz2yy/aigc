"""
代理，可以让 LLM 拥有使用工具的能力！
目前支持的工具：https://python.langchain.com/en/latest/modules/agents/tools.html
可以自定义工具：https://python.langchain.com/en/latest/modules/agents/tools/custom_tools.html
    例子：visual GPT: https://github.com/microsoft/visual-chatgpt
"""

from langchain.agents import load_tools, get_all_tool_names
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json
# 可以在环境变量设置 openai key，或者通过参数传入
# llm = OpenAI(temperature=0, openai_api_key="YourAPIKey")
# llm = OpenAI(temperature=0)

# 所有内置支持的工具
print(get_all_tool_names())

# 需要注册一个 serpapi 账号，https://serpapi.com/manage-api-key
serpapi_api_key = "..."

toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)

agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)

response = agent({"input": "what was the first album of the" "band that Natalie Bergman is a part of?"})

