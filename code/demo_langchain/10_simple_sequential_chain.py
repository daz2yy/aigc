"""
线性处理链，避免同时执行多任务的时候让 LLM 分心、混乱
需求：输入一个地址，给出当地经典美食，并给出在家制作的方法
分两步处理：
1. 获取用户所在地的经典美食
2. 根据第一步产生的结果请求获取在家制作的方法
"""

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# 可以在环境变量设置 openai key，或者通过参数传入
# llm = OpenAI(temperature=1, openai_api_key="YourAPIKey")
llm = OpenAI(temperature=1)

# 1. 获取用户所在地的经典美食
template = """Your job is to come up with a classic dish from the area that the users suggests. % USER LOCATION {user_location} YOUR RESPONSE: """
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Holds my 'location' chain
location_chain = LLMChain(llm=llm, prompt=prompt_template)

# 2. 根据第一步产生的结果请求获取在家制作的方法
template = """Given a meal, give a short and simple recipe on how to make that dish at home. % MEAL {user_meal} YOUR RESPONSE: """
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=llm, prompt=prompt_template)

# 串联1、2两步; verbose=True 是可以把生成的过程打印出来，方便调试
overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)

# 测试结果
review = overall_chain.run("Rome")

"""
结果：
> Entering new SimpleSequentialChain chain...


Spaghetti alla Carbonara- a classic Roman dish made with spaghetti, guanciale, egg, Parmigiano-Reggiano, and black pepper.


Ingredients: 
- 8 ounces of spaghetti
- 2 large eggs
- 33 grams of guanciale (or pancetta) 
- 2 tablespoons of Parmigiano-Reggiano
- Freshly ground black pepper to taste

Instructions: 
1. Cook the spaghetti in well-salted boiling water until al dente.

2. Meanwhile, heat a pan over medium heat and add the guanciale. Cook until lightly browned, about 5 minutes.

3. Beat the eggs in a bowl and stir in the freshly grated Parmigiano-Reggiano and black pepper.

4. Drain the spaghetti, reserving a bit of the cooking liquid. Add the spaghetti to the pan with the guanciale.

5. Remove pan from heat and stir in the egg and cheese mixture. If the spaghetti is too dry, add a little of the cooking liquid.

6. Serve immediately. Enjoy!

> Finished chain.
"""