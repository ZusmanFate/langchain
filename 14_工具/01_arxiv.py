""" 
本文件是【工具和工具箱：LangChain 中的 Tool 和 Toolkits 一览】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388071002242023462
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
# 设置OpenAI API的密钥
import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
# 导入库
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

# 初始化模型和工具
llm = ChatOpenAI(
    temperature=0.0,
    model=os.environ.get("LLM_MODELEND"),
)
tools = load_tools(
    ["arxiv"],
)

# 初始化链
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 运行链
agent_chain.run("介绍一下2005.14165这篇论文的创新点?")
