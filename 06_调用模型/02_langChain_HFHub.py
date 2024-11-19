""" 
本文件是【调用模型：使用OpenAI API还是微调开源Llama2/ChatGLM？】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7396297142727344168
由于网络限制，Huggingface.co无法正常访问，请考虑使用国内镜像站点下载模型后、本地加载模型使用。
"""
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 导入必要的库
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# 初始化HF LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    # repo_id="meta-llama/Llama-2-7b-chat-hf",
)

# 创建简单的question-answering提示模板
template = """Question: {question}
              Answer: """

# 创建Prompt
prompt = PromptTemplate(template=template, input_variables=["question"])

# 调用LLM Chain --- 我们以后会详细讲LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 准备问题
question = "Rose is which type of flower?"

# 调用模型并返回结果
print(llm_chain.run(question))
