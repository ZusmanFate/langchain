import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
# 导入所需库
from langchain import PromptTemplate, LLMChain
from langchain_openai import ChatOpenAI

# 设置提示模板
prompt = PromptTemplate(
    input_variables=["flower", "season"], template="{flower}在{season}的花语是?"
)

# 初始化大模型
llm = ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0)

# 初始化链
llm_chain = LLMChain(llm=llm, prompt=prompt)

# 调用链
response = llm_chain({"flower": "玫瑰", "season": "夏季"})
print(response)

# run方法
llm_chain.run({"flower": "玫瑰", "season": "夏季"})

# predict方法
result = llm_chain.predict(flower="玫瑰", season="夏季")
print(result)

# apply方法允许您针对输入列表运行链
input_list = [
    {"flower": "玫瑰", "season": "夏季"},
    {"flower": "百合", "season": "春季"},
    {"flower": "郁金香", "season": "秋季"},
]
result = llm_chain.apply(input_list)
print(result)

# generate方法
result = llm_chain.generate(input_list)
print(result)
