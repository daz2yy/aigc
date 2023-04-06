"""
把文本切分成小的文本块，并分别总结文本块的内容，最后把所有的总结结合在一起总结得出最终的总结
例子生成的结果貌似有点问题
"""
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 可以在环境变量设置 openai key，或者通过参数传入
# llm = OpenAI(temperature=1, openai_api_key="YourAPIKey")
llm = OpenAI(temperature=1)

loader = TextLoader("data/阿Q正传.txt")

documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts)



