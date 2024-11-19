import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
# 导入所需的库
from langchain import PromptTemplate, LLMChain
from langchain_openai import ChatOpenAI

# 原始字符串模板
template = "{flower}的花语是?"
# 创建模型实例
llm = ChatOpenAI(temperature=0, model=os.environ.get("LLM_MODELEND"))
# 创建LLMChain
llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template))
# 调用LLMChain，返回结果
result = llm_chain("玫瑰")
print(result)
