"""
检索器，把输入的文本转化为向量，并与已有的数据内容对比获得最相似的结果并返回
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader("data/阿Q正传.txt")
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
# 可以在环境变量设置 openai key，或者通过参数传入
# embeddings = OpenAIEmbeddings(openai_api_key="YourAPIKey")
embeddings = OpenAIEmbeddings()

# Embedd your texts
db = FAISS.from_documents(texts, embeddings)

# Init your retriever. Asking for just 1 document back
retriever = db.as_retriever()

# 查询
docs = retriever.get_relevant_texts("阿Q是什么样的人？")

print("\n\n".join([x.page_content[:200] for x in docs[:2]]))
