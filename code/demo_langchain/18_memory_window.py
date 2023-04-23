from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    memory=memory,
    verbose=True
)
# llm_chain.predict(human_input="你是谁？")

def ask_bot(input_index = 'index.json'):
    while True:
        query = input('What do you want to ask the bot?   \n')
        response = llm_chain.predict(human_input=query)
        print ("\nBot says: \n\n" + response + "\n\n\n")

ask_bot()


