"""
通过 VectorDBQA 来实现先搜索再回复的能力
zh_core_web_sm 中文分词模型
en_core_web_sm 英文分词模型
python -m spacy download zh_core_web_sm en_core_web_sm
"""

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader

llm = OpenAI(temperature=0)
loader = TextLoader('./data/CS_问答集_BP_0329_prepared.txt')
documents = loader.load()
# text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
text_splitter = SpacyTextSplitter(chunk_size=256)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

faq_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=docsearch, verbose=True)

# 使用
question = "How Can I refund?"
result = faq_chain.run(question)
print(result)


