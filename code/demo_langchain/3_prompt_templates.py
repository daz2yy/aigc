"""
非常有用，可用于动态生成不同的 prompt
langchain 内置了一些有用的变量，比如生成的内容 content，可以设置模版为根据生成内容做处理等
"""
from langchain.llms import OpenAI
from langchain import PromptTemplate

# 可以在环境变量设置 openai key，或者通过参数传入
# llm = OpenAI(model_name="text-davinci-003", openai_api_key="YourAPIKey")
llm = OpenAI(model_name="text-davinci-003")

# 注意 {location} 这个变量，后续会被替换成不同的值
template = """ I really want to travel to {location}. What should I do there? Respond in one short sentence """

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location="Rome")

print(f"Final Prompt: {final_prompt}")
print("-----------")
print(f"LLM Output: {llm(final_prompt)}")
"""
结果：
Final Prompt:  I really want to travel to Rome. What should I do there? Respond in one short sentence 
-----------
LLM Output: 

Explore the ancient ruins and vibrant culture of Rome.
"""
