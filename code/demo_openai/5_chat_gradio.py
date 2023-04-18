"""
需求：
- 使用 Gradio 开发界面
    - python 框架
    - 可以直接在 Jupyter Notebook 里面显示出来
    - 被 HuggingFace 收购，开源
- 部署在 HuggingFace 上
优化：
1. 你能根据这一讲学到的内容，修改一下代码，让这个聊天机器人不限制轮数，只在 Token 数量要超标的时候再删减最开始的对话么？
2. 除了“忘记”开始的几轮，你还能想到什么办法，让 AI 尽可能多地记住上下文么？
TODO：显示有问题
"""
import openai
import gradio as gr

prompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内"""


class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]
        num_of_tokens = response['usage']['total_tokens']
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round * 2 + 1:
            del self.messages[1:3]  # Remove the first round conversation left.
        return message, num_of_tokens


conv = Conversation(prompt, 10)


def answer(question, history=[]):
    history.append(question)
    response = conv.ask(question)
    history.append(response)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history


with gr.Blocks(css="#chatbot{height:300px} .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    txt.submit(answer, [txt, state], [chatbot, state])

demo.launch()
