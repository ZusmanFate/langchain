""" 
本文件是【记忆：通过 Memory 记住客户上次买花时的对话细节】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388070989826883621
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
# 设置OpenAI API密钥
import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
# 导入所需的库
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# 初始化大语言模型
llm = ChatOpenAI(
    model=os.environ.get("LLM_MODELEND"),
    temperature=0.5,
)

# 初始化对话链
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

# 第一天的对话
# 回合1
conversation("我姐姐明天要过生日，我需要一束生日花束。")
print("第一次对话后的记忆:", conversation.memory.buffer)

# 回合2
conversation("她喜欢粉色玫瑰，颜色是粉色的。")
print("第二次对话后的记忆:", conversation.memory.buffer)

# 回合3 （第二天的对话）
conversation("我又来了，还记得我昨天为什么要来买花吗？")
print("/n第三次对话后时提示:/n", conversation.prompt.template)
print("/n第三次对话后的记忆:/n", conversation.memory.buffer)
