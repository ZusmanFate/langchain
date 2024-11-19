# 设置OpenAI API密钥
import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"

# 导入所需要的库
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd

# 定义 LLM 模型
llm = ChatOpenAI(
    temperature=0.7,
    model=os.environ.get("LLM_MODELEND"),
)

# 第一个链的提示模板：生成鲜花的介绍
template_intro = """
你是一个植物学家。给定花的名称和颜色，你需要为这种花写一个200字左右的介绍。
花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:"""
prompt_intro = PromptTemplate(input_variables=["name", "color"], template=template_intro)
introduction_chain = LLMChain(
    llm=llm, prompt=prompt_intro, output_key="introduction"
)

# 第二个链的提示模板：根据鲜花的介绍写评论
template_review = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。
鲜花介绍:
{introduction}
花评人对上述花的评论:"""
prompt_review = PromptTemplate(input_variables=["introduction"], template=template_review)
review_chain = LLMChain(
    llm=llm, prompt=prompt_review, output_key="review"
)

# 第三个链的提示模板：根据介绍和评论写社交媒体帖子
template_post = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}
社交媒体帖子:
"""
prompt_post = PromptTemplate(input_variables=["introduction", "review"], template=template_post)
social_post_chain = LLMChain(
    llm=llm, prompt=prompt_post, output_key="social_post_text"
)

# 总链：按顺序运行三个链
overall_chain = SequentialChain(
    chains=[introduction_chain, review_chain, social_post_chain],
    input_variables=["name", "color"],
    output_variables=["introduction", "review", "social_post_text"],
    verbose=True,
)

# 结构化输出解析器
response_schemas = [
    ResponseSchema(name="introduction", description="鲜花介绍："),
    ResponseSchema(name="review", description="花评人对上述花的评论:"),
    ResponseSchema(name="social_post_text", description="社交媒体帖子:"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# 创建 DataFrame
df = pd.DataFrame(columns=["flower", "color", "introduction", "review", "social_post_text"])

# 循环遍历每种花和颜色
flowers = ["玫瑰", "百合", "康乃馨"]
colors = ["黑色", "白色", "粉色"]

for flower, color in zip(flowers, colors):
    # 运行总链
    output = overall_chain({"name": flower, "color": color})

    # 解析输出
    parsed_output = {
        "flower": flower,
        "color": color,
        "introduction": output["introduction"],
        "review": output["review"],
        "social_post_text": output["social_post_text"]
    }

    # 添加到 DataFrame
    df = pd.concat([df, pd.DataFrame([parsed_output])], ignore_index=True)

# 打印 DataFrame
print(df.to_dict(orient="records"))
