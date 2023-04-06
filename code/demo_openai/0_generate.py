"""
调用 OpenAI 的 Completion 接口，然后向它提了一个需求，为一个我在 1688 上找到的中文商品名称做三件事情。
1. 为这个商品写一个适合在亚马逊上使用的英文标题。
2. 给这个商品写 5 个卖点。
3. 估计一下，这个商品在美国卖多少钱比较合适。
同时，告诉 OpenAI，我们希望返回的结果是 JSON 格式的，并且上面的三个事情用 title、selling_points 和 price_range 三个字段返回。
"""
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

prompt = """Consideration proudct : 工厂现货PVC充气青蛙夜市地摊热卖充气玩具发光蛙儿童水上玩具

1. Compose human readable product title used on Amazon in english within 20 words.
2. Write 5 selling points for the products in Amazon.
3. Evaluate a price range for this product in U.S.

Output the result in json format with three properties called title, selling_points and price_range"""

def get_response(prompt):
    completions = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.0,
    )
    message = completions.choices[0].text
    return message

print(get_response(prompt))

"""
返回的信息结构：
{
    "title": "Factory Stock PVC Inflatable Frog Night Market Hot Selling Inflatable Toy Glowing Frog Water Toy for Kids",
    "selling_points": [
        "Made of high-quality PVC material, safe and durable",
        "Inflatable design, easy to store and carry",
        "Glow in the dark, bring more fun to kids",
        "Perfect for pool, beach, lake, etc.",
        "Ideal gift for kids"
    ],
    "price_range": "$10 - $20"
}
"""
