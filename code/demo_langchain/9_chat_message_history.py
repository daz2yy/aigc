"""
聊天历史记录器（本质上，发送给 OpenAI 的数据都是字符串，Langchain 只是做了字符串的处理封装，不用开发者再去处理数据格式）
"""

from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

# 可以在环境变量设置 openai key，或者通过参数传入
chat = ChatOpenAI(temperature=0)

history = ChatMessageHistory()

history.add_ai_message("hi!")

history.add_user_message("what is the capital of france?")

# llm 处理请求
ai_response = chat(history.messages)

# 保存新的记录
history.add_ai_message(ai_response.content)

print(history.messages)
