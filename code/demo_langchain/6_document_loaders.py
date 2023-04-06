"""
文档加载器，可以让用户把不同的数据源加载到程序中（举例，爬取网页信息，直接获取到最终数据的那一部分，不需要从0开始编写）
"""
from langchain.document_loaders import HNLoader

loader = HNLoader("https://news.ycombinator.com/item?id=34422627")

data = loader.load()

print(f"Found {len(data)} comments")
print(f"Here's a sample:\n\n{''.join([x.page_content[:150] for x in data[:2]])}")

"""
结果：
Found 76 comments
Here's a sample:

Ozzie_osman 78 days ago  
             | next [–] 

LangChain is awesome. For people not sure what it's doing, large language models (LLMs) are very pOzzie_osman 78 days ago  
             | parent | next [–] 

Also, another library to check out is GPT Index (https://github.com/jerryjliu/gpt_index) 

"""