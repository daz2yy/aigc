"""
基本的 prompt 用法
"""
from langchain.llms import OpenAI

# 可以在环境变量设置 openai key，或者通过参数传入
# llm = OpenAI(model_name="text-davinci-003")
llm = OpenAI(model_name="text-davinci-003", openai_api_key="YourAPIKey")

# I like to use three double quotation marks for my prompts because it's easier to read
prompt = """ Today is Monday, tomorrow is Wednesday. What is wrong with that statement? """

print(llm(prompt))
