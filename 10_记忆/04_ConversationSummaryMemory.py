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
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

# 初始化大语言模型
llm = ChatOpenAI(
    temperature=0.5,
    model=os.environ.get("LLM_MODELEND"),
)

# 初始化对话链
conversation = ConversationChain(
    llm=llm, memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=300)
)

# 第一天的对话
# 回合1
result = conversation("我姐姐明天要过生日，我需要一束生日花束。")
print(result)
# 回合2
result = conversation("她喜欢粉色玫瑰，颜色是粉色的。")
# print("\n第二次对话后的记忆:\n", conversation.memory.buffer)
print(result)

# 第二天的对话
# 回合3
result = conversation("我又来了，还记得我昨天为什么要来买花吗？")
print(result)
