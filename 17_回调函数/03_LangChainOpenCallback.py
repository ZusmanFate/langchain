import os

from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain_core.messages.human import HumanMessage
import asyncio

# 初始化大语言模型
llm = ChatOpenAI(
    model=os.environ["LLM_MODELEND"],
    temperature=0.5,
)

# 初始化对话链
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

# 使用context manager进行token counting
with get_openai_callback() as cb:
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

# 输出使用的tokens
print("\n总计使用的tokens:", cb.total_tokens)


# 进行更多的异步交互和token计数
async def additional_interactions():
    with get_openai_callback() as cb:
        await asyncio.gather(
            *[
                llm.agenerate(
                    messages=[[HumanMessage(content="我姐姐喜欢什么颜色的花？")]]
                )
                for _ in range(3)
            ]
        )
    print("\n另外的交互中使用的tokens:", cb.total_tokens)


# 运行异步函数
asyncio.run(additional_interactions())
