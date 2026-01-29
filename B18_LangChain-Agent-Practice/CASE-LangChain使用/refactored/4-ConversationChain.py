from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from app_config import llm

# 创建带历史记录的 prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 创建 chain
chain = prompt | llm

# 存储会话历史
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建带记忆的对话链
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "default"}}

# 第一轮对话
output = conversation.invoke({"input": "Hi there, I am wesley!"}, config=config)
print(output.content)

# 第二轮对话 (会记住上一轮)
output = conversation.invoke({"input": "Hi Buddy, do you remember my name ?."}, config=config)
print(output.content)
