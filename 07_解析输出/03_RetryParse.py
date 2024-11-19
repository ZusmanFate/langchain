template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""

# 定义一个Pydantic数据格式，这个格式描述了一个"行动"类及其属性
from pydantic import BaseModel, Field

class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


# 使用Pydantic格式Action来初始化一个输出解析器
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Action)

# 定义一个提示模板，它将用于向模型提问
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="What are the colors of Orchid?")

# 定义一个错误格式的字符串
bad_response = '{"action": "search"}'
# parser.parse(bad_response) # 如果直接解析，它会引发一个错误

# 设置OpenAI API密钥
import os
from langchain_openai import ChatOpenAI
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
# 尝试用OutputFixingParser来解决这个问题
from langchain.output_parsers import OutputFixingParser

fix_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=ChatOpenAI(
        model=os.environ.get("LLM_MODELEND"),
    ),
)
parse_result = fix_parser.parse(bad_response)
print("OutputFixingParser的parse结果:", parse_result)

# 初始化RetryWithErrorOutputParser，它会尝试再次提问来得到一个正确的输出
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser, llm=ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0)
)
parse_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print("RetryWithErrorOutputParser的parse结果:", parse_result)
