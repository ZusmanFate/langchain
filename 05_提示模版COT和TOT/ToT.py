# 设置环境变量和API密钥
import os

os.environ["OPENAI_API_KEY"] = '3d8947f7-cbd9-4482-a0f7-ac266d429da0'

# 创建聊天模型
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.environ.get("LLM_MODELEND"),
)

# 设定 AI 的角色和目标
role_template = (
    "你是一个为旅游电商公司工作的AI助手，负责帮助客户选择最佳的旅行目的地和行程安排。"
)

# ToT 的关键部分，AI 根据用户的需求逐层筛选，最终给出推荐并解释理由
tot_template = """
作为一个为旅游电商公司工作的AI助手，我的目标是帮助客户根据他们的需求选择最佳的旅行方案。

我的推理过程如下：

1. 理解客户的需求，包括预算、旅行时间、喜爱的活动等。
2. 根据预算筛选出符合要求的目的地。
3. 根据季节和时长筛选出气候合适的目的地。
4. 考虑客户的活动喜好，并找到活动丰富的地方。
5. 为用户提供推荐，并详细解释我的推荐理由。

示例：
  人类：我想找一个适合春天的旅行地点，预算在5000元左右，喜欢户外活动。
  AI：首先，我理解你需要一个适合春天的户外旅行地点，预算为5000元。根据这一点，我推荐以下几个选择。
  推荐1：云南大理。春天的大理气候温暖，户外活动丰富，适合徒步和观光。住宿费用和餐饮都在你的预算范围内，是一个理想的选择。
"""

# 构建系统和用户的提示模板
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
system_prompt_tot = SystemMessagePromptTemplate.from_template(tot_template)
# 用户的询问
human_template = "{human_input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages(
    [system_prompt_role,system_prompt_tot, human_prompt]
)

# 示例用户需求
prompt = chat_prompt.format_prompt(
    human_input="我想找一个适合夏季的旅行地点，预算大约在7000元，喜欢历史文化景点。有什么推荐吗？"
).to_messages()

# 接收用户的询问，返回结果
response = llm(prompt)
print(response)
