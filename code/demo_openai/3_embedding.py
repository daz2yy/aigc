"""
openai embedding 测试
数据下载地址：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset
- 今日头条中文新闻（文本）分类数据集
- 方便做对比
(数据比较多，会消耗很多费用）

机器学习的指标：
1. 准确率（Precision）
- 模型判断属于这个分类的数据有多少是真正属于这个分类的；比如模型判断有100个属于这个分类，但实际上100个里面只有80个是真的属于这个分类，那么准确率就只有80%
2. 召回率（Recall）
- 模型判断属于这个分类的数据在这个分类下所有数据的占比；比如模型判断有100个属于这个分类，但实际上这个分类有200个数据，那召回率就只有50%
3. F1 分数
- 准确率和召回率的调和平均数，F1 Score = 2 / (1/Precision + 1/Recall)；分数越高越好
4. 支持样本量（Support）
- 分类的数据有多少
5. accuracy
- 样本整体的准确率，所有预测准确的数据 / 总数据量
6. macro average
- 宏平均，每个分类的数据平均
7. weighted average
- 加权平均，按照样本量加权平均

注意：
1. embedding 计算 token 的时候注意编码格式，否则可能出现和 openai 计算不一致的问题
2. text-embedding-ada-002 限制最大 token 是 8191
3. openai 有接口限速，具体查看：https://platform.openai.com/docs/guides/rate-limits/overview
- 编码是注意用接口请求的回避算法，比如 backoff
4. openai 支持 batch 请求
5. 大数据集，不要存储成 CSV 格式, 存储成 CSV 格式会把本来只需要 4 个字节的浮点数，都用字符串的形式存储下来，会浪费好几倍的空间，写入的速度也很慢
"""

import pandas as pd
import tiktoken
import openai
import os
import backoff

from openai.embeddings_utils import get_embedding, get_embeddings

openai.api_key = os.environ.get("OPENAI_API_KEY")
use_batch = False

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# import data/toutiao_cat_data.txt as a pandas dataframe
df = pd.read_csv('data/toutiao_cat_data.txt', sep='_!_', names=['id', 'code', 'category', 'title', 'keywords'])
df = df.fillna("")
df["combined"] = (
        "标题: " + df.title.str.strip() + "; 关键字: " + df.keywords.str.strip()
)

print("Lines of text before filtering: ", len(df))

encoding = tiktoken.get_encoding(embedding_encoding)
# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens]

print("Lines of text after filtering: ", len(df))

# 增加 backoff 退避算法
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embedding_with_backoff(**kwargs):
    return get_embedding(**kwargs)

# 增加 openai batch 请求
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings


if not use_batch:
    # randomly sample 10k rows
    df_10k = df.sample(10000, random_state=42)

    df_10k["embedding"] = df_10k.combined.apply(lambda x : get_embedding_with_backoff(text=x, engine=embedding_model))
    df_10k.to_csv("data/toutiao_cat_data_10k_with_embeddings.csv", index=False)
else:
    batch_size = 1000
    # randomly sample 10k rows
    df_all = df
    # group prompts into batches of 100
    prompts = df_all.combined.tolist()
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    embeddings = []
    for batch in prompt_batches:
        batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
        embeddings += batch_embeddings

    df_all["embedding"] = embeddings
    df_all.to_parquet("data/toutiao_cat_data_all_with_embeddings.parquet", index=True)

