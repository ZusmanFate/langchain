import os
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel
from volcenginesdkarkruntime import Ark
from langchain_openai import ChatOpenAI  # ChatOpenAI模型

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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    class Config:
        arbitrary_types_allowed = True


class ChatbotWithRetrieval:
    def __init__(self, dir):

        # 加载Documents
        base_dir = dir  # 文档的存放目录
        documents = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".docx") or file.endswith(".doc"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        # 文本的分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

        # 向量数据库
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits,  # 以分块的文档
            embedding=DoubaoEmbeddings(
                model=os.environ["EMBEDDING_MODELEND"],
            ),  # 用OpenAI的Embedding Model做嵌入
            location=":memory:",  # in-memory 存储
            collection_name="my_documents",
        )  # 指定collection_name

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=os.environ["LLM_MODELEND"],
            temperature=0,
        )

        # 初始化Memory
        self.memory = ConversationSummaryMemory(
            llm=self.llm, memory_key="chat_history", return_messages=True
        )
        # 初始化对话历史
        self.conversation_history = ""

        # 设置Retrieval Chain
        retriever = self.vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=retriever, memory=self.memory
        )

    def get_response(self, user_input):  # 这是为 Gradio 创建的新函数
        response = self.qa(user_input)
        # 更新对话历史
        self.conversation_history += (
            f"你: {user_input}\nChatbot: {response['answer']}\n"
        )
        return self.conversation_history


if __name__ == "__main__":
    folder = "23_Chatbot下/OneFlower"
    bot = ChatbotWithRetrieval(folder)

    # 定义 Gradio 界面
    interface = gr.Interface(
        fn=bot.get_response,  # 使用我们刚刚创建的函数
        inputs="text",  # 输入是文本
        outputs="text",  # 输出也是文本
        live=False,  # 实时更新，这样用户可以连续与模型交互
        title="易速鲜花智能客服",  # 界面标题
        description="请输入问题，然后点击提交。",  # 描述
    )
    interface.launch(share=True)  # 启动 Gradio 界面
