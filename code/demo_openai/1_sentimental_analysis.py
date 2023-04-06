"""
需求：使用 LLM 进行情感分析

传统的二分类方法：朴素贝叶斯与逻辑回归
- 情感分析原理
    - 当成一个分类问题
    - 把一组人工标注过好坏的数据用于模型训练
    - 用另一部分未标注的数据进行测试（或者是用一部分标注过的数据测试，这样直观的对比模型进行预测分类的结果的有效性如何）
- 朴素贝叶斯
    - 一个单词在好评出现的概率比在差评里高的多，那这个词所在的评论就有可能是一个差评
传统方法的挑战：特征工程与模型调参
- 特征工程
    - 比如，不但是看这个单词出现的概率，还关联上前后几个单词出现的概率
        - 实现：2-Gram（Bigram 双字节词组）和 3-Gram（Trigram 三字节词组）
    - 比如，去掉停用词，去掉过于低频的词等
- 模型调参
    - 数据集切分成训练（Training）、验证（Validation）、测试（Test）三组数据
    - 通过 AUC 或者混淆矩阵（Confusion Matrix）来衡量效果
    - 如果数据量不够多，为了训练效果的稳定性，可能需要采用 K-Fold 的方式来进行训练。

LLM 解决方法：
- 通过 LLM 计算好评、差评的向量距离
- 计算需要测试的文本向量距离
- 用余弦距离（曼哈顿距离）计算以上两者的相似性，得出结果更靠进哪一边（和好评的距离 - 和差评的距离，正数就靠近好评，负数是靠近差评）

"""


import openai
import os
from openai.embeddings_utils import cosine_similarity, get_embedding

# 获取访问open ai的密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
# 选择使用最小的ada模型
EMBEDDING_MODEL = "text-embedding-ada-002"

# 获取"好评"和"差评"的
positive_review = get_embedding("好评")
negative_review = get_embedding("差评")

positive_example = get_embedding("买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质")
negative_example = get_embedding("降价厉害，保价不合理，不推荐")

def get_score(sample_embedding):
    return cosine_similarity(sample_embedding, positive_review) - cosine_similarity(sample_embedding, negative_review)

positive_score = get_score(positive_example)
negative_score = get_score(negative_example)

print("好评例子的评分 : %f" % (positive_score))
print("差评例子的评分 : %f" % (negative_score))

"""
返回结果：
好评例子的评分 : 0.070963
差评例子的评分 : -0.072895
"""
