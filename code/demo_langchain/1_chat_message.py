"""
可以设置AI角色，AI 通过上下文生成文本
"""
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# 可以在环境变量设置 openai key，或者通过参数传入
chat = ChatOpenAI(temperature=0.7)
ai_message = chat([SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
      HumanMessage(content="I like tomatoes, what should I eat?")])

print(ai_message)

"""
结果；
content='You could try a Caprese salad with fresh tomatoes, mozzarella cheese, and basil.' additional_kwargs={}
"""
