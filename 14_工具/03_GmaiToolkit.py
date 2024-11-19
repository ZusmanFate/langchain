import os
os.environ["OPENAI_API_KEY"] = "3d8947f7-cbd9-4482-a0f7-ac266d429da0"
# 导入与Gmail交互所需的工具包
from langchain.agents.agent_toolkits import GmailToolkit

# 初始化Gmail工具包
toolkit = GmailToolkit()

# 从gmail工具中导入一些有用的功能
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials

# 获取 Gmail API 的凭证，并指定相关的权限范围
# 访问 Gmail 服务的代码可能无法正常运行
credentials = get_gmail_credentials(
    token_file="token.json",  # Token文件路径
    scopes=["https://mail.google.com/"],  # 具有完全的邮件访问权限
    client_secrets_file="credentials.json",  # 客户端的秘密文件路径
)
# 使用凭证构建API资源服务
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# 获取工具
tools = toolkit.get_tools()
print(tools)

# 导入与聊天模型相关的包
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# 初始化聊天模型
llm = ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0)

# 通过指定的工具和聊天模型初始化agent
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

# 使用agent运行一些查询或指令
result = agent.run("今天易速鲜花客服给我发邮件了么？最新的邮件是谁发给我的？")

# 打印结果
print(result)
