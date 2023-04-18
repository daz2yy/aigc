"""
https://time.geekbang.org/column/article/646363
使用开源模型的向量搜索替代 llama-index 向量搜索部分
"""
import openai, os
import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTFaissIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser

openai.api_key = ""

# chunk_size 设置为100，因为所使用的开源模型是个小模型，这样我们才能在单机加载起来
# chunk_overlap 设置为20，这个参数代表我们自动合并小的文本片段的时候，可以接受多大程度的重叠
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
parser = SimpleNodeParser(text_splitter=text_splitter)
documents = SimpleDirectoryReader('./data').load_data()
nodes = parser.get_nodes_from_documents(documents)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
))
service_context = ServiceContext.from_defaults(embed_model=embed_model)

dimension = 768
faiss_index = faiss.IndexFlatIP(dimension)
index = GPTFaissIndex(nodes=nodes, faiss_index=faiss_index, service_context=service_context)

# 测试效果
from llama_index import QueryMode

openai.api_key = os.environ.get("OPENAI_API_KEY")

response = index.query(
    "How can i refund?",
    mode=QueryMode.EMBEDDING,
    verbose=True,
)
print(response)
