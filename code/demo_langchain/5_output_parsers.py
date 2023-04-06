"""
输出内容格式化，把 LLM 生成的数据进行格式化处理；比如，返回一个 json 结构的结果
"""
import json

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.llms import OpenAI

# 可以在环境变量设置 openai key，或者通过参数传入
# llm = OpenAI(model_name="text-davinci-003", openai_api_key="YourAPIKey")
llm = OpenAI(model_name="text-davinci-003")

# How you would like your reponse structured. This is basically a fancy prompt template

response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response"),
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# See the prompt template you created for formatting
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# 生成数据
template = (
    """ You will be given a poorly formatted string from a user. Reformat it and make sure all the words are spelled correctly {format_instructions} % USER INPUT: {user_input} YOUR RESPONSE: """
)

prompt = PromptTemplate(input_variables=["user_input"], partial_variables={"format_instructions": format_instructions}, template=template)

promptValue = prompt.format(user_input="welcom to califonya!")

print(promptValue)

response = llm(promptValue)
print("============= response: ", response)

# 返回 json 数据
print("============= parse:", output_parser.parse(response))
"""
最后的生成结果：
```json
{
	"bad_string": "welcom to califonya!"
	"good_string": "Welcome to California!"
}
```
"""