import asyncio
import os
import re
import jieba

from langchain_openai import ChatOpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

# ==========================================
# 1. 全局配置
# ==========================================
# 用 Ollama 拉取专门用于 Embedding 的高分轻量模型 (速度极快)
# ollama pull bge-m3
Settings.embed_model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")
langchain_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", temperature=0)
Settings.llm = LangChainLLM(llm=langchain_llm)
# 👉 新增：调大分块上限，让单份常规文档能一次性被塞入一个 Chunk 中
Settings.chunk_size = 2048
# 👉 新增：增加分块之间的重叠字数，防止极端长文被切断时丢失上下文
Settings.chunk_overlap = 200

DOCS_DIR = "./docs"
DB_PATH = "./milvus_local.db"
BM25_MODEL_PATH = "./bm25_model.json"
COLLECTION_NAME = "my_hybrid_knowledge_base"


# ==========================================
# 2. 中文 BM25 核心逻辑包装器
# ==========================================
# 声明使用官方内置的中文分词器（底层会自动调用 jieba）
official_analyzer = build_default_analyzer(language="zh")
raw_bm25_ef = BM25EmbeddingFunction(analyzer=official_analyzer)


class LlamaIndexBM25Wrapper(BaseSparseEmbeddingFunction):
    def __init__(self, bm25_ef: BM25EmbeddingFunction):
        self.bm25_ef = bm25_ef

    def encode_queries(self, queries: list[str]) -> list[dict]:
        return self._to_dict_list(self.bm25_ef.encode_queries(queries))

    def encode_documents(self, documents: list[str]) -> list[dict]:
        return self._to_dict_list(self.bm25_ef.encode_documents(documents))

    def _to_dict_list(self, matrix) -> list[dict]:
        # 1. 强制转换为 CSR 格式，确保底层拥有完整的三个指针数组
        matrix = matrix.tocsr()
        results = []
        # 2. 直接根据内存指针跨度，极速提取每一行的数据（100% 免疫版本差异）
        for i in range(matrix.shape[0]):
            start_idx = matrix.indptr[i]
            end_idx = matrix.indptr[i + 1]

            indices = matrix.indices[start_idx:end_idx]
            data = matrix.data[start_idx:end_idx]

            results.append({int(k): float(v) for k, v in zip(indices, data)})
        return results


wrapped_bm25_ef = LlamaIndexBM25Wrapper(raw_bm25_ef)


# ==========================================
# 3. 数据清洗与向量库构建; rm -f ./milvus_local.db ./milvus_local.db-wal ./milvus_local.db-shm ./bm25_model.json
# ==========================================
def clean_text(text: str) -> str:
    if not text or not isinstance(text, str): return ""
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

    # 👉 修改：只压缩连续的水平空格，但保留换行符 \n
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)  # 将3个以上连续空行压缩为2个

    # 👉 修改：在正则结尾加入了 \n，避免其被当做非法字符清洗掉
    text = re.sub(r'[^\w\s\u4E00-\u9FFF\(\)（）\[\]【】\{\}，。？！；：\-_\.，、；：？！\n]', ' ', text)
    return text.strip() if text.strip() else "查询词"


def validate_documents(documents: list) -> list:
    validated_docs = []
    for doc in documents:
        if isinstance(doc, Document):
            cleaned = clean_text(doc.text)
            if len(cleaned) < 5: continue
            # 修复最新版 LlamaIndex 中 Document.text 为只读的问题
            new_doc = Document(text=cleaned, metadata=doc.metadata, id_=doc.id_)
            validated_docs.append(new_doc)
        else:
            validated_docs.append(doc)
    return validated_docs


async def get_or_build_milvus_index():
    print(f"📂 正在扫描 '{DOCS_DIR}' 目录...")

    if os.path.exists(DB_PATH) and os.path.exists(BM25_MODEL_PATH):
        print(f"📦 发现本地数据库与中文词频库，正在加载权重...")
        raw_bm25_ef.load(BM25_MODEL_PATH)
        vector_store = MilvusVectorStore(
            uri=DB_PATH, collection_name=COLLECTION_NAME, dim=1024,
            enable_sparse=True, sparse_embedding_function=wrapped_bm25_ef,
            hybrid_ranker="RRFRanker", hybrid_ranker_params={"k": 60}
        )
        return VectorStoreIndex.from_vector_store(vector_store)

    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    if not documents: return None

    documents = validate_documents(documents)
    if not documents: return None

    print("🧠 正在使用 jieba 训练专属的中文 BM25 词频库...")
    corpus = [doc.text for doc in documents]
    raw_bm25_ef.fit(corpus)
    raw_bm25_ef.save(BM25_MODEL_PATH)

    vector_store = MilvusVectorStore(
        uri=DB_PATH, collection_name=COLLECTION_NAME, dim=1024,
        enable_sparse=True, sparse_embedding_function=wrapped_bm25_ef,
        hybrid_ranker="RRFRanker", hybrid_ranker_params={"k": 60}
    )

    print(f"⏳ 开始向量化，写入 Milvus Lite '{DB_PATH}'...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)


# ==========================================
# 4. 主程序
# ==========================================
async def main():
    index = await get_or_build_milvus_index()
    if not index: return

    print("🚀 正在配置中文混合检索引擎 (Jieba关键字 + BGE向量语义)...")
    query_engine = index.as_query_engine(vector_store_query_mode="hybrid", similarity_top_k=3)
    vector_only_engine = index.as_query_engine(vector_store_query_mode="default", similarity_top_k=3)

    print("-" * 50)
    while True:
        question = input("\n🙋 请输入你的问题 (输入 q 退出): ")
        if question.lower() == 'q': break

        question = clean_text(question)
        if not question or question == "查询词": continue

        print("🤖 云端大脑思考中...")
        try:
            response = query_engine.query(question)
            print(f"👉 回答:\n{response}\n")
        except Exception as e:
            print(f"⚠️ 混合检索失败: {e}，切换向量检索...")
            try:
                print(f"👉 回答 (向量模式):\n{vector_only_engine.query(question)}\n")
            except Exception as e2:
                print(f"❌ 失败：{e2}")


if __name__ == "__main__":
    asyncio.run(main())