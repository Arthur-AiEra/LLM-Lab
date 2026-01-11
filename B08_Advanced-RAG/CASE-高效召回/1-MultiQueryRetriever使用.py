from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from typing import List
import os

# 获取环境变量中的 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)

# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 加载向量数据库
vectorstore = FAISS.load_local("./vector_db", embeddings, allow_dangerous_deserialization=True)

def generate_multi_queries(query: str, llm, num_queries: int = 3) -> List[str]:
    """使用LLM生成多个查询变体"""
    prompt = f"""你是一个AI助手，负责生成多个不同视角的搜索查询。
给定一个用户问题，生成{num_queries}个不同但相关的查询，以帮助检索更全面的信息。

原始问题: {query}

请直接输出{num_queries}个查询，每行一个，不要编号和其他内容:"""

    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    return [query] + queries[:num_queries]

def multi_query_search(query: str, vectorstore, llm, k: int = 4) -> List:
    """执行多查询检索，合并去重结果"""
    queries = generate_multi_queries(query, llm)
    print(f"生成的查询变体: {queries}")

    seen_contents = set()
    unique_docs = []

    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

    return unique_docs

# 示例查询
query = "客户经理的考核标准是什么？"
# 执行查询
results = multi_query_search(query, vectorstore, llm)

# 打印结果
print(f"\n查询: {query}")
print(f"找到 {len(results)} 个相关文档:")
for i, doc in enumerate(results):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
