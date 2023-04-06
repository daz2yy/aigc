"""
文本向量化，把文本转化为一维浮点数组的形式
"""
from langchain.embeddings import OpenAIEmbeddings

# 可以在环境变量设置 openai key，或者通过参数传入
# embeddings = OpenAIEmbeddings(openai_api_key="YourAPIKey")
embeddings = OpenAIEmbeddings()

text = "Hi! It's time for the beach"

text_embedding = embeddings.embed_query(text)
print(f"Your embedding is length {len(text_embedding)}")
print(f"Here's a sample: {text_embedding[:5]}...")
