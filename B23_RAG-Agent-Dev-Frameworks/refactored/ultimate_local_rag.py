import asyncio
import os
import re
# 👇 新增这三行代码：强制屏蔽底层 gRPC 的 C++ 烦人日志 👇
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
os.environ["GLOG_minloglevel"] = "2" # 屏蔽 absl 等 C++ 库的 INFO/WARNING 日志

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document,
    load_index_from_storage  # 👉 新增：用于加载本地持久化的索引状态
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
import logging

# --- 1. 环境配置与日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. 全局配置
# ==========================================
Settings.embed_model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")

# 👉 换脑手术：接入你刚下载的 Qwen3.5 MoE 模型
Settings.llm = Ollama(
    model="qwen3.5:35b-a3b",
    base_url="http://localhost:11434",
    request_timeout=360.0,  # 给本地大模型留出足够的思考时间
    temperature=0.1         # 保持回答严谨不发散
)

# 👉 新增：调大分块上限，让单份常规文档能一次性被塞入一个 Chunk 中
Settings.chunk_size = 2048
# 👉 新增：增加分块之间的重叠字数，防止极端长文被切断时丢失上下文
Settings.chunk_overlap = 200

DOCS_DIR = "./docs"
DB_PATH = "ultimate_local_rag/milvus_local.db"
BM25_MODEL_PATH = "ultimate_local_rag/bm25_model.json"
PERSIST_DIR = "ultimate_local_rag/storage"  # 👉 新增：存放文档 Hash 记录表的目录
COLLECTION_NAME = "ultimate_local_knowledge_base"

# ==========================================
# 2. 中文 BM25 核心逻辑包装器
# ==========================================
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
        matrix = matrix.tocsr()
        results = []
        for i in range(matrix.shape[0]):
            start_idx = matrix.indptr[i]
            end_idx = matrix.indptr[i + 1]
            indices = matrix.indices[start_idx:end_idx]
            data = matrix.data[start_idx:end_idx]
            results.append({int(k): float(v) for k, v in zip(indices, data)})
        return results


wrapped_bm25_ef = LlamaIndexBM25Wrapper(raw_bm25_ef)


# ==========================================
# 3. 数据清洗与向量库构建 (含增量更新魔法)
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

            # 👉 增量更新核心：提取固定的绝对路径 + 页码作为永久 ID，防止每次随机生成 UUID
            file_path = doc.metadata.get("file_path", doc.id_)
            page_label = doc.metadata.get("page_label", "1")
            doc_id = f"{file_path}_page_{page_label}"

            new_doc = Document(text=cleaned, metadata=doc.metadata, id_=doc_id)
            validated_docs.append(new_doc)
        else:
            validated_docs.append(doc)
    return validated_docs


async def get_or_build_milvus_index():
    print(f"📂 正在扫描 '{DOCS_DIR}' 目录...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    if not documents: return None

    documents = validate_documents(documents)
    if not documents: return None
    print(f"✅ 共发现 {len(documents)} 份有效文档片段。")

    # [BM25 词频库加载/训练]
    if os.path.exists(BM25_MODEL_PATH):
        print(f"📦 发现本地中文词频库，正在加载权重...")
        raw_bm25_ef.load(BM25_MODEL_PATH)
    else:
        print("🧠 正在使用 jieba 训练专属的中文 BM25 词频库...")
        corpus = [doc.text for doc in documents]
        raw_bm25_ef.fit(corpus)
        raw_bm25_ef.save(BM25_MODEL_PATH)

    # 初始化 Milvus 引擎
    vector_store = MilvusVectorStore(
        uri=DB_PATH, collection_name=COLLECTION_NAME, dim=1024,
        enable_sparse=True, sparse_embedding_function=wrapped_bm25_ef,
        hybrid_ranker="RRFRanker", hybrid_ranker_params={"k": 60}
    )

    # 👉 增量更新逻辑分支
    if os.path.exists(PERSIST_DIR) and os.path.exists(DB_PATH):
        print(f"📦 发现本地记录 '{PERSIST_DIR}' 和向量库，正在加载历史索引状态...")
        # 组装 StorageContext，既包含 Milvus 向量库，又包含本地 Hash 缓存目录
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=PERSIST_DIR
        )
        index = load_index_from_storage(storage_context)

        print("🔄 正在比对新旧文档，执行增量更新检查...")
        refreshed_status = index.refresh_ref_docs(documents)
        new_or_updated_count = sum(refreshed_status)

        if new_or_updated_count > 0:
            print(f"✨ 增量更新完毕！已重新向量化 {new_or_updated_count} 份新文档/修改内容。")
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print("💾 最新索引状态已保存。")
        else:
            print("👍 没有发现新文档，知识库已经是最新状态，秒开完成！")

        return index

    else:
        print(f"🚀 首次运行：正在进行全局首次向量化，写入 Milvus Lite '{DB_PATH}'...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        # 👉 极其关键：首次运行完毕后，必须将文档 Hash 状态持久化到本地目录，否则下次无法对比
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"💾 首次向量化完成！文档 Hash 状态已保存至 '{PERSIST_DIR}'。")
        return index


def print_source_references(response, mode_name="混合检索"):
    """封装打印检索来源的逻辑"""
    if not response.source_nodes:
        return

    print(f"📚 检索到的参考文件来源 ({mode_name}):")
    for i, node in enumerate(response.source_nodes, 1):
        # 从 metadata 中提取文件路径和页码
        file_path = node.metadata.get("file_path", "未知文件路径")
        page_label = node.metadata.get("page_label", "未知页码")
        # 提取得分并保留4位小数
        score = round(node.score, 4) if node.score is not None else "无"

        print(f"  [{i}] 文件: {file_path} (页码: {page_label}) | 检索得分: {score}")
    print("-" * 50)

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
        question = input("\n🙋 请输入你的问题 (输入 q 退出): ") # 雇主安心保和雇主责任险，两者的赔偿范围有哪些不同？平安境内紧急医疗救援服务的责任免除有哪些？
        if question.lower() == 'q': break

        question = clean_text(question)
        if not question or question == "查询词": continue

        print("🤖 本地 Qwen3.5 大脑正在结合 Milvus 混合检索结果进行思考...")
        try:
            response = query_engine.query(question)
            logging.info(f"👉 回答:\n{response}\n")
            # 👉 调用封装的方法，打印混合检索来源
            print_source_references(response, mode_name="混合检索")
        except Exception as e:
            print(f"⚠️ 混合检索失败: {e}，切换向量检索...")
            try:
                # 降级尝试向量检索
                fallback_response = vector_only_engine.query(question)
                print(f"👉 回答 (向量模式):\n{fallback_response}\n")
                # 👉 调用封装的方法，打印向量检索来源
                print_source_references(fallback_response, mode_name="向量模式")
            except Exception as e2:
                print(f"❌ 失败：{e2}")


if __name__ == "__main__":
    asyncio.run(main())