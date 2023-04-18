"""
https://time.geekbang.org/column/article/646363
使用清华的 ChatGLM 模型
最大的模型有1300亿个参数，本地跑不起来，选一个 60亿参数的模型来跑
"""
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
# cuda() 使用 GPU 跑模型
# trust_remote_code 因为不是 Huggingface 官方发布的，需要加上这个参数表示信任
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

# CPU 运行模型
# model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4",trust_remote_code=True).float()

# 测试模型

question = """
自收到商品之日起7天内，如产品未使用、包装完好，您可以申请退货。某些特殊商品可能不支持退货，请在购买前查看商品详情页面的退货政策。

根据以上信息，请回答下面的问题：

Q: 你们的退货政策是怎么样的？
"""
response, history = model.chat(tokenizer, question, history=[])
print(response)