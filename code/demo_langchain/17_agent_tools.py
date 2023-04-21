"""
让 LangChain 使用工具解决问题
Agent的工作：
1. Action，就是根据用户的输入，选择应该选取哪一个 Tool，然后行动。
2. Action Input，就是根据需要使用的 Tool，从用户的输入里提取出相关的内容，可以输入到 Tool 里面。
3. Oberservation，就是观察通过使用 Tool 得到的一个输出结果。
4. Thought，就是再看一眼用户的输入，判断一下该怎么做。
5. Final Answer，就是 Thought 在看到 Obersavation 之后，给出的最终输出。
源码：https://github.com/hwchase17/langchain/blob/master/langchain/agents/mrkl/prompt.py
zero-shot-react-description 这个想法来源于 AI21 Labs论文：https://arxiv.org/pdf/2205.00445.pdf
"""

import openai, os

openai.api_key = os.environ.get("OPENAI_API_KEY")

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# llm = ChatOpenAI(max_tokens=2048, temperature=0.5)
# multiple_choice = """
# 请针对 >>> 和 <<< 中间的用户问题，选择一个合适的工具去回答她的问题。只要用A、B、C的选项字母告诉我答案。
# 如果你觉得都不合适，就选D。
#
# >>>{question}<<<
#
# 我们有的工具包括：
# A. 一个能够查询商品信息，为用户进行商品导购的工具
# B. 一个能够查询订单信息，获得最新的订单情况的工具
# C. 一个能够搜索商家的退换货政策、运费、物流时长、支付渠道、覆盖国家的工具
# D. 都不合适
# """
# multiple_choice_prompt = PromptTemplate(template=multiple_choice, input_variables=["question"])
# choice_chain = LLMChain(llm=llm, prompt=multiple_choice_prompt, output_key="answer")
#
# # 测试
# question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
# print(choice_chain(question))
#
# question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
# print(choice_chain(question))
#
# question = "请问你们的货，能送到三亚吗？大概需要几天？"
# print(choice_chain(question))
#
# question = "今天天气怎么样？"
# print(choice_chain(question))
#
# # Tools 例子
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期：2023-01-01；预计送达时间：2023-01-10"

def recommend_product(input: str) -> str:
    return "红色连衣裙"

def faq(intput: str) -> str:
    return "7天无理由退货"
#
# tools = [
#     Tool(
#         name = "Search Order",func=search_order,
#         description="useful for when you need to answer questions about customers orders"
#     ),
#     Tool(name="Recommend Product", func=recommend_product,
#          description="useful for when you need to answer questions about product recommendations"
#          ),
#     Tool(name="FAQ", func=faq,
#          description="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."
#          )
# ]
# # ReAct 来自：https://ai.googleblog.com/2022/11/react-synergizing-reasoning-and-acting.html，并非 facebook 的前端框架
# # verbose 调试开关；max_iterations 重试思考的次数
# agent = initialize_agent(tools, llm, max_iterations=2, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# # 测试
# question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
# result = agent.run(question)
# print(result)


# =========> 通过向量数据查询支持 Tool 回答
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import VectorDBQA

llm = OpenAI(temperature=0)
loader = TextLoader('./data/CS_问答集_BP_0329_prepared.txt')
documents = loader.load()
# text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
text_splitter = SpacyTextSplitter(chunk_size=1500)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

faq_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=docsearch, verbose=True)

# 使用
question = "How Can I refund?"
result = faq_chain.run(question)
print(result)


from langchain.agents import tool


# 通过 @tool 这个 Python 的 decorator 功能，将 FAQ 这个函数直接变成了 Tool 对象，这可以减少我们每次创建 Tools 的时候都要指定 name 和 description 的工作。
@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about beautyplus app, like return refund, function usage, etc."""
    return faq_chain.run(intput)


tools = [
    Tool(
        name="Search Order", func=search_order,
        description="useful for when you need to answer questions about customers orders"
    ),
    Tool(name="Recommend Product", func=recommend_product,
         description="useful for when you need to answer questions about product recommendations"
         ),
    faq
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

question = "请问 iOS 如何退款？"
result = agent.run(question)
print(result)
