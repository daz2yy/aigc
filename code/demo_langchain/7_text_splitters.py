"""
可以把文档分割为更小的块（chunk），以便于模型更高效的处理
chunk_overlap, 分割成块的时候是否可以重叠，重叠多少次
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

# This is a long document we can split up.
with open("data/阿Q正传.txt") as f:
    pg_work = f.read()

print(f"You have {len([pg_work])} document")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a tiny chunk size, just to show.
    chunk_size=150,
    chunk_overlap=20,
)

texts = text_splitter.create_documents([pg_work])

print(f"You have {len(texts)} documents")

print("Preview:")
print(texts[0].page_content, "\n")
print(texts[1].page_content)

"""
结果：
You have 1 document
You have 222 documents
Preview:
第一章 序 

我要给阿q做正传，已经不止一两年了。但一面要做，一面又往回想，这足见我不是一个“立言”的人，因为从来不朽之笔，须传不朽之人，于是人以文传，文以人传——究竟谁靠谁传，渐渐的不甚了然起来，而终于归接到传阿q，仿佛思想里有鬼似的。

"""