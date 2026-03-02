import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_openai import ChatOpenAI
from llama_index.llms.langchain import LangChainLLM


# ==========================================
# 1. 全局配置：本地 Embedding + 云端 LLM
# ==========================================
Settings.embed_model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")
langchain_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)
Settings.llm = LangChainLLM(llm=langchain_llm)

PERSIST_DIR = "./local_storage"
DOCS_DIR = "./docs"


def load_and_refresh_index():
    """带增量更新逻辑的索引加载器"""

    print(f"📂 正在扫描 '{DOCS_DIR}' 目录下的最新文档...")
    # 每次运行都读取当前目录下的所有文件状态
    current_documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    for doc in current_documents:
        # 使用文件的绝对路径作为固定 ID
        file_path = doc.metadata.get("file_path", doc.id_)
        # 考虑到如果是 PDF 可能会按页切分，加上页码避免 ID 冲突覆盖
        page_label = doc.metadata.get("page_label", "1")
        # 强制将随机 UUID 替换为稳定的结构化 ID
        doc.id_ = f"{file_path}_page_{page_label}"
    print(f"✅ 共发现 {len(current_documents)} 份文档片段。")

    if os.path.exists(PERSIST_DIR):
        print(f"📦 发现本地缓存 '{PERSIST_DIR}'，正在加载历史索引...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

        print("🔄 正在比对新旧文档，执行增量更新检查...")

        # 核心魔法：refresh_ref_docs 会自动对比文档的 Hash 值
        # 只有新增或被修改过的文档，才会重新走本地 Embedding 算力
        refreshed_status = index.refresh_ref_docs(current_documents)

        # refreshed_status 是一个布尔值列表 (如 [False, False, True, ...])
        # True 代表该文档是新的或被修改过，已重新向量化
        new_or_updated_count = sum(refreshed_status)

        if new_or_updated_count > 0:
            print(f"✨ 增量更新完毕！为你处理了 {new_or_updated_count} 份新文档/修改内容。")
            # 既然知识库有了新内容，务必再次保存到本地硬盘
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print("💾 最新索引已保存。")
        else:
            print("👍 没有发现新文档，知识库已经是最新状态，秒开完成！")

        return index

    else:
        print("🚀 首次运行：正在进行全局首次向量化 (这可能需要一些时间)...")
        index = VectorStoreIndex.from_documents(current_documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"💾 首次向量化完成！索引已保存至 '{PERSIST_DIR}'。")
        return index


def main():
    # 获取索引（自动完成秒开或增量更新）
    index = load_and_refresh_index()

    # 转化为查询引擎
    query_engine = index.as_query_engine(similarity_top_k=3)

    print("-" * 50)
    while True:
        question = input("🙋 请输入你的问题 (输入 q 退出): ") # 平安商业综合责任保险（亚马逊）的理赔流程是？
        if question.lower() == 'q':
            break

        print("🤖 云端大脑正在结合本地检索结果进行思考...")
        response = query_engine.query(question)
        print(f"👉 回答:\n{response}\n")


if __name__ == "__main__":
    main()