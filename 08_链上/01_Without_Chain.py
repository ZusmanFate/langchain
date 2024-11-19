""" 
本文件是【链（上）：写一篇完美鲜花推文？用SequencialChain链接不同的组件】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388070991978561588
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
# ----第一步 创建提示
# 导入LangChain中的提示模板
from langchain import PromptTemplate

# 原始字符串模板
template = "{flower}的花语是?"
# 创建LangChain模板
prompt_temp = PromptTemplate.from_template(template)
# 根据模板创建提示
prompt = prompt_temp.format(flower="玫瑰")
# 打印提示的内容
print(prompt)

# ----第二步 创建并调用模型
# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI

# 创建模型实例
model = ChatOpenAI(temperature=0, model=os.environ.get("LLM_MODELEND"))
# 传入提示，调用模型，返回结果
result = model.predict(prompt)
print(result)
