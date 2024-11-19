""" 
本文件是【代理（下）：结构化工具对话、Self-Ask with Search 以及 Plan and execute 代理】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388070985528213538
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
# 设置OpenAIAPI密钥
import os

from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser


async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
print(tools)

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI  # ChatOpenAI模型

# LLM不稳定，对于这个任务，可能要多跑几次才能得到正确结果
llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


async def main():
    response = await agent_chain.arun("What are the headers on python.langchain.com?")
    print(response)


import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
