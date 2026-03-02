import os

from langchain_openai import ChatOpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

#https://gemini.google.com/app/d7cccf1e1a25a5b6
# 用 Ollama 拉取专门用于 Embedding 的高分轻量模型 (速度极快)
# ollama pull bge-m3
# localhost:11434

# ==========================================
# 1. 全局配置：本地 Embedding + 云端 LLM
# ==========================================
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434"
)
langchain_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)
# 将 LangChain LLM 封装为 LlamaIndex LLM
Settings.llm = LangChainLLM(llm=langchain_llm)

DOCS_DIR = "./docs"
ES_INDEX_NAME = "my_knowledge_base"

def get_or_build_es_index():
    """连接 Elasticsearch 并管理索引"""

    print("🔌 正在连接本地 Elasticsearch (http://localhost:9200)...")
    # 初始化 Elasticsearch 向量存储
    es_store = ElasticsearchStore(
        index_name=ES_INDEX_NAME,
        es_url="http://localhost:9200",
        retrieval_strategy=AsyncDenseVectorStrategy()  # 使用密集向量检索（不需要 RRF 许可证）
    )

    # 将 ES 绑定到 LlamaIndex 的存储上下文中
    storage_context = StorageContext.from_defaults(vector_store=es_store)

    # 简单的逻辑判定：这里为了演示，我们每次启动读取一遍文档
    # (在实际生产中，ES 作为外部数据库是持久化的。你可以直接 VectorStoreIndex.from_vector_store(es_store) 来加载已存在的库)
    print(f"📂 正在扫描 '{DOCS_DIR}' 目录下的文档...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()

    if len(documents) > 0:
        print(f"⏳ 开始向量化并将数据写入 Elasticsearch 索引 '{ES_INDEX_NAME}'...")
        # 这一步会将文本、向量、以及元数据一并存入 ES，构建出既能做向量对比、又能做倒排索引的混合库
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True  # 在终端显示向量化进度条
        )
        print("✅ 成功写入 Elasticsearch！")
        return index
    else:
        # 如果没有新文档，直接连接已有的 ES 库
        print("⏭️ 未发现文档，直接挂载已存在的 Elasticsearch 索引。")
        return VectorStoreIndex.from_vector_store(es_store)


def main():
    # 1. 初始化或连接 ES 索引
    index = get_or_build_es_index()

    # 2. 核心魔法：开启密集向量检索模式
    print("🚀 正在配置混合检索引擎 (BM25 关键字 + 向量语义)....")
    query_engine = index.as_query_engine(
        vector_store_query_mode="default",  # 使用默认密集向量检索（跳过 RRF 许可证要求）
        similarity_top_k=3,  # 返回最相关的 3 个片段
    )

    print("-" * 50)
    # 3. 交互式问答循环
    while True:
        question = input("\n🙋 请输入你的问题 (输入 q 退出): ")
        if question.lower() == 'q':
            break

        print("🤖 云端大脑正在结合 ES 混合检索结果进行思考...")

        # 提问时，底层实际上向 ES 发起了两个并发查询：一个算余弦距离，一个算 BM25 词频
        response = query_engine.query(question)

        print(f"👉 回答:\n{response}\n")

        # 进阶调试：你可以打印出到底是哪些文档片段被召回了，看看混合检索的威力
        # print("\n[召回参考源]:")
        # for node in response.source_nodes:
        #     print(f" - 文件: {node.metadata.get('file_name')} | 分数: {node.score}")


if __name__ == "__main__":
    main()