""" 
本文件是【代理（上）：ReAct 框架，推理与行动的协同】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388070996097368116
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
# 设置OpenAI和SERPAPI的API密钥
import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
os.environ["SERPAPI_API_KEY"] = (
    "32c2066b72804a61e0b3d69b8dd05d3fd50b36063c6573d4ee16580222c857a7"
)

# 加载所需的库
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI  # ChatOpenAI模型

# 初始化大模型
llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

# 设置工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化Agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# 跑起来
agent.run(
    "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"
)
