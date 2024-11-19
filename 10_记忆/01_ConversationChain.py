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

# 初始化大语言模型
llm = ChatOpenAI(
    model=os.environ.get("LLM_MODELEND"),
    temperature=0.5,
)

# 初始化对话链
conv_chain = ConversationChain(llm=llm)

# 打印对话的模板
print(conv_chain.prompt.template)
