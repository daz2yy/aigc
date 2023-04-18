"""
1. 基于 Embedding 向量进行文本聚类
2. 摘要

知识点：
文本聚类: 就是把很多没有标注过的文本，根据它们之间的相似度，自动地分成几类。
    - 因为我们把文本变成了向量，所以比较向量相似性就可以了，算法有 K-means
    - 常用数据集：20 newsgroups
"""


import pandas as pd
import openai
import os
import backoff
from openai.embeddings_utils import get_embeddings
from sklearn.datasets import fetch_20newsgroups

# 获取新闻数据（数据在sklearn库里）
def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame( newsgroups_train.target_names, columns=['title'])

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out.to_csv('data/20_newsgroup.csv', index=False)

# twenty_newsgroup_to_csv()


# 过滤掉数据里面有些文本是空的情况
# 把 Token 数量太多的给过滤掉

# import openai, os, tiktoken, backoff
#
openai.api_key = os.environ.get("OPENAI_API_KEY")
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

df = pd.read_csv('data/20_newsgroup.csv')
print("Number of rows before null filtering:", len(df))
df = df[df['text'].isnull() == False]
encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
print("Number of rows before token number filtering:", len(df))
df = df[df.n_tokens <= max_tokens]
print("Number of rows data used:", len(df))


# 获取 embedding
batch_size = 2000
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings

prompts = df.text.tolist()
prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

embeddings = []
for batch in prompt_batches:
    batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
    embeddings += batch_embeddings

df["embedding"] = embeddings
df.to_parquet("data/20_newsgroup_with_embedding.parquet", index=False)


# K-Means 算法计算聚类

import numpy as np
from sklearn.cluster import KMeans

embedding_df = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

matrix = np.vstack(embedding_df.embedding.values)
num_of_clusters = 20

kmeans = KMeans(n_clusters=num_of_clusters, init="k-means++", n_init=10, random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_
embedding_df["cluster"] = labels

# 聚类完，我们怎么去看它聚类的结果是不是合适呢？每个聚合出来的类代表什么呢？
# 我们统计一下聚类之后的每个类有多少条各个 newsgroups 分组的数据。然后看看这些数据里面，排名第一的分组是什么。如果我们聚类聚合出来的类，都是从某一个 newsgroup 分组出来的文章，那么说明这个聚合出来的类其实就和那个分组的内容差不多。

# 统计每个cluster的数量
new_df = embedding_df.groupby('cluster')['cluster'].count().reset_index(name='count')

# 统计这个cluster里最多的分类的数量
title_count = embedding_df.groupby(['cluster', 'title']).size().reset_index(name='title_count')
first_titles = title_count.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
first_titles = first_titles.reset_index(drop=True)
new_df = pd.merge(new_df, first_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
new_df = new_df.rename(columns={'title': 'rank1', 'title_count': 'rank1_count'})

# 统计这个cluster里第二多的分类的数量
second_titles = title_count[~title_count['title'].isin(first_titles['title'])]
second_titles = second_titles.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
second_titles = second_titles.reset_index(drop=True)
new_df = pd.merge(new_df, second_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
new_df = new_df.rename(columns={'title': 'rank2', 'title_count': 'rank2_count'})
new_df['first_percentage'] = (new_df['rank1_count'] / new_df['count']).map(lambda x: '{:.2%}'.format(x))
# 将缺失值替换为 0
new_df.fillna(0, inplace=True)
# 输出结果
display(new_df)


# 让 OpenAI 生成类目摘要


items_per_cluster = 1
COMPLETIONS_MODEL = "text-davinci-003"

for i in range(num_of_clusters):
    cluster_name = new_df[new_df.cluster == i].iloc[0].rank1
    print(f"Cluster {i}, Rank 1: {cluster_name}, 抽样翻译:", end=" ")

    content = "\n".join(
        embedding_df[(embedding_df.cluster == i) & (embedding_df.n_tokens > 100)].text.sample(items_per_cluster, random_state=42).values
    )
    response = openai.Completion.create(
        model=COMPLETIONS_MODEL,
        prompt=f'''请把下面的内容翻译成中文\n\n内容:\n"""\n{content}\n"""翻译：''',
        temperature=0,
        max_tokens=2000,
        top_p=1,
    )
    print(response["choices"][0]["text"].replace("\n", ""))
