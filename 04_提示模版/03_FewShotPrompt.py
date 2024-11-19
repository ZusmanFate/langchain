samples = [
    {
        "flower_type": "玫瑰",
        "occasion": "爱情",
        "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。",
    },
    {
        "flower_type": "康乃馨",
        "occasion": "母亲节",
        "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。",
    },
    {
        "flower_type": "百合",
        "occasion": "庆祝",
        "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。",
    },
    {
        "flower_type": "向日葵",
        "occasion": "鼓励",
        "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。",
    },
]

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# 2. 创建一个提示模板
from langchain.prompts.prompt import PromptTemplate

prompt_sample = PromptTemplate(
    input_variables=["flower_type", "occasion", "ad_copy"],
    template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}",
)
print(prompt_sample.format(**samples[0]))

# 3. 创建一个FewShotPromptTemplate对象
from langchain.prompts.few_shot import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=samples,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"],
)
print(prompt.format(flower_type="野玫瑰", occasion="爱情"))

# 4. 把提示传递给大模型
import os

os.environ["OPENAI_API_KEY"] = '3d8947f7-cbd9-4482-a0f7-ac266d429da0' # API Key
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model=os.environ.get("LLM_MODELEND"),
)
result = model(prompt.format(flower_type="野玫瑰", occasion="爱情"))
print(result)

# 5. 使用示例选择器
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

# 初始化Embedding类
from volcenginesdkarkruntime import Ark
from typing import List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel


class DoubaoEmbeddings(BaseModel, Embeddings):
    client: Ark = None
    api_key: str = ""
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.api_key == "":
            self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = Ark(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=self.api_key
        )

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(model=self.model, input=text)
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    class Config:
        arbitrary_types_allowed = True


# 初始化示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    DoubaoEmbeddings(
        model=os.environ.get("EMBEDDING_MODELEND"),
    ),
    Chroma,
    k=1,
)

# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"],
)
print(prompt.format(flower_type="红玫瑰", occasion="爱情"))
