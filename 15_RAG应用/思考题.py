import os
from volcenginesdkarkruntime import Ark
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Any
# 自定义文本嵌入类
# 初始化Embedding类
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


# 加载文档并分割文本
loader = TextLoader("15_RAG应用/OneFlower/花语大全.txt", encoding="utf8")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = loader.load()
split_documents = text_splitter.split_documents(documents)

# 使用Chroma向量数据库存储嵌入
embeddings = DoubaoEmbeddings(model=os.environ["EMBEDDING_MODELEND"])
vectorstore = Chroma.from_documents(split_documents, embeddings)

# 使用RetrievalQA检索器进行查询
llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# 查询
query = "玫瑰花的花语是什么？"
result = qa_chain.run(query)

# 打印结果
print(result)
