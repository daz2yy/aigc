
Langchain 主要的组件: Schema、Models、Prompts、Indexes、Memory、Chains、Agents

# Schema
- Text
  - 最基本的文本字符串
  ```
    # You'll be working with simple strings (that'll soon grow in complexity!)
    my_text = "What day comes after Friday?"
  ```
- Chat Message
  - openai 的 Chat 封装
- Documents
  - 文本片段，附带 metadata 信息
  ```
  Document(
    page_content="This is my document. It is full of text that I've gathered from other places",
    metadata={"my_document_id": 234234, "my_document_source": "The LangChain Papers", "my_document_create_time": 1680013019},
  )
  ```
## Models
- Language Models
  - 语言模型，输入文字，输出文字
- Chat Models
  - 聊天模型，可以设定AI的角色来聊天
- Text Embedding Models
  - 把文本转化成数字向量，可以方便的对比和分析文本，比如比较两个文本的相似性确定内容含义的相似度；一维浮点数组形式

## Prompts
- Basic Prompt
  - 和语言模型的基本用法一致，输入文本，输出文本
- Prompt Templates
  - 非常有用，可用于动态生成不同的 prompt；举例，遇到不同的人说不同的话
- Example Selectors
  - 根据用户的输入，选择相似的 example 提供给 AI 作为参考，然后去回答问题
  - 使用到了 FewShotPromptTemplate，用少数的例子生成答案
  - 可能比较适合于分类场景
- Output Parsers
  - 输出内容格式化，把 LLM 生成的数据进行格式化处理；比如，返回一个 list 结构的结果

## Indexes
- Document Loaders
  - 文档加载器，可以让用户把不同的数据源加载到程序中（举例，爬取网页信息，直接获取到最终数据的那一部分，不需要从0开始编写）
- Text Splitters
  - 可以把文档分割为更小的块（chunk），以便于模型更高效的处理
- Retrievers
  - 检索器，把输入的文本转化为向量，并与已有的数据内容对比获得最相似的结果并返回
- VectorStores
  - 通过向量来存储和搜索目标内容。向量是用数字数组来表示语义。
  - 向量数据库：Pinecone、Weaviate、

## Memory
- Chat Message History
  - 聊天历史记录器（本质上，发送给 OpenAI 的数据都是字符串，Langchain 只是做了字符串的处理封装，不用开发者再去处理数据格式）

## Chains: 链，支持组合多个 LLM 调用执行
- Simple Sequential Chain
  - 线性处理链，避免同时执行多任务的时候让 LLM 分心、混乱
- Summarization Chain
  - 把文本切分成小的文本块，并分别总结文本块的内容，最后把所有的总结结合在一起总结得出最终的总结

## Agents：代理，可以让 LLM 拥有使用工具的能力！
- 


## 资料
[A Comprehensive Guide to LangChain](https://nathankjer.com/introduction-to-langchain/?utm_source=bensbites&utm_medium=newsletter&utm_campaign=new-match-found-open-source-funding)

