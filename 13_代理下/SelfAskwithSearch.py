""" 
本文件是【代理（下）：结构化工具对话、Self-Ask with Search 以及 Plan and execute 代理】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388070985528213538
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
# 设置OpenAI和SERPAPI的API密钥
import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
os.environ["SERPAPI_API_KEY"] = (
    "32c2066b72804a61e0b3d69b8dd05d3fd50b36063c6573d4ee16580222c857a7"
)
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

llm = ChatOpenAI(
    model=os.environ["LLM_MODELEND"],
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.run("使用玫瑰作为国花的国家的首都是哪里?")
