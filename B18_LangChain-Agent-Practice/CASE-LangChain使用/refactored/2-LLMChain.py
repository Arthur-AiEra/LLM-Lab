from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools

from app_config import llm

# SERPAPI_API_KEY， https://serpapi.com/


# 加载 serpapi 工具
tools = load_tools(["serpapi"])

# LangChain 1.x 新写法
agent = create_agent(llm, tools)

# 运行 agent
# result = agent.invoke({"messages": [("user", "今天是几月几号?历史上的今天有哪些名人出生")]})
result = agent.invoke({"messages": [("user", "特朗普和委内瑞拉最近有什么新闻")]})
print(result["messages"][-1].content)
