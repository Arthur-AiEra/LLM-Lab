from langchain_openai import ChatOpenAI
import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
