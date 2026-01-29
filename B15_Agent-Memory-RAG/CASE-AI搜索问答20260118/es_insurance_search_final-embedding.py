import os
from elasticsearch import Elasticsearch
from pathlib import Path
import fitz  # PyMuPDF for PDF files
import docx  # python-docx for DOCX files
import numpy as np
from openai import OpenAI

# Elasticsearch连接配置
ES_HOST = 'localhost'
ES_PORT = 9200
ES_USERNAME = 'elastic'
ES_PASSWORD = '5A7C1+=PbQCpkw1jvu-8'

def get_embedding(text, dimensions=1024):
    """使用text-embedding-v4获取文本嵌入向量"""
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        completion = client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=dimensions,
            encoding_format="float"
        )
        return completion.data[0].embedding
    except Exception as e:
        print(f"获取嵌入向量时出错: {e}")
        # 返回零向量作为备选
        return [0.0] * dimensions

def extract_text_from_pdf(file_path):
    """从PDF文件中提取文本内容"""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    """从DOCX文件中提取文本内容"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path):
    """从TXT文件中提取文本内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as file:
                return file.read()
        except:
            print(f"[WARN] 无法读取文件 {file_path}")
            return ""
    except Exception as e:
        print(f"Error reading TXT file {file_path}: {e}")
        return ""

def get_document_content(file_path):
    """根据文件扩展名提取文档内容"""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    elif extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file type: {extension}")
        return ""

def split_document_into_chunks(content, chunk_size=512):
    """将文档内容分割成小块"""
    words = content.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
    
    return chunks

def index_documents_to_es():
    """连接到Elasticsearch，创建带向量字段的索引，索引文档"""
    # 连接到Elasticsearch
    es_url = f"https://{ES_HOST}:{ES_PORT}"
    try:
        es = Elasticsearch(
            es_url,
            basic_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # 检查连接
        if es.ping():
            print("[SUCCESS] 成功连接到 Elasticsearch")
        else:
            print("无法连接到Elasticsearch")
            return None
    except Exception as e:
        print(f"连接到Elasticsearch时发生错误: {e}")
        print("请确保Elasticsearch服务已在 https://localhost:9200 启动")
        return None
    
    # 定义索引名称
    index_name = 'insurance_documents_embeddings'
    
    # 删除已存在的索引（如果存在）
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"已删除现有索引: {index_name}")
    
    # 创建新的索引 - 包含向量字段
    mapping = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "file_path": {
                    "type": "keyword"
                },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 1024,  # text-embedding-v4的维度
                    "index": True,
                    "similarity": "cosine"  # 使用余弦相似度
                },
                "chunk_id": {
                    "type": "integer"
                }
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, body=mapping)
        print(f"[SUCCESS] 成功创建索引 {index_name} (包含向量字段)")
    except Exception as e:
        print(f"创建索引时发生错误: {e}")
        return None
    
    # 获取docs目录中的所有文件
    docs_dir = Path('./docs')
    if not docs_dir.exists():
        print("docs目录不存在")
        return
    
    # 计数器
    total_files = 0
    indexed_chunks = 0
    
    for file_path in docs_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
            print(f"[INFO] 处理文件: {file_path.name}")
            content = get_document_content(file_path)
            total_files += 1
            
            if content.strip():  # 确保文档内容不为空
                # 将文档分割成块
                chunks = split_document_into_chunks(content)
                
                for chunk_id, chunk_content in enumerate(chunks):
                    # 获取文档块的嵌入向量
                    vector = get_embedding(chunk_content)
                    
                    doc = {
                        "title": file_path.name,
                        "content": chunk_content,
                        "file_path": str(file_path.absolute()),
                        "content_vector": vector,  # 添加向量字段
                        "chunk_id": chunk_id
                    }
                    
                    try:
                        es.index(index=index_name, body=doc)
                        indexed_chunks += 1
                        
                        # 每索引10个块打印一次进度
                        if indexed_chunks % 10 == 0:
                            print(f"[PROGRESS] 已索引 {indexed_chunks} 个文档块")
                            
                    except Exception as e:
                        print(f"[ERROR] 索引文档块时出错: {e}")
            else:
                print(f"[WARN] 跳过空文档: {file_path.name}")
    
    # 刷新索引以使文档立即可用
    es.indices.refresh(index=index_name)
    print(f"[SUCCESS] 总共处理了 {total_files} 个文件，成功索引 {indexed_chunks} 个文档块")
    
    return es, index_name

def search_in_documents(es, index_name, search_query):
    """在Elasticsearch中使用向量搜索文档"""
    # 首先获取查询的嵌入向量
    query_vector = get_embedding(search_query)
    
    # 构建向量搜索查询
    query_body = {
        "knn": {
            "field": "content_vector",
            "query_vector": query_vector,
            "k": 10,  # 返回最相似的10个结果
            "num_candidates": 100  # 搜索的候选数量
        },
        "_source": ["title", "content", "file_path", "chunk_id"],  # 返回的字段
        "highlight": {
            "fields": {
                "content": {
                    "fragment_size": 150,
                    "number_of_fragments": 3
                }
            }
        }
    }
    
    try:
        # 执行搜索
        response = es.search(index=index_name, body=query_body)
        
        # 显示搜索结果
        print(f"\n搜索查询: {search_query}")
        print(f"使用向量搜索找到 {response['hits']['total']['value']} 个结果:\n")
        
        if response['hits']['total']['value'] > 0:
            print("="*80)
            
            for i, hit in enumerate(response['hits']['hits'], 1):
                source = hit['_source']
                title = source.get('title', '无标题')
                file_path = source.get('file_path', '未知路径')
                chunk_id = source.get('chunk_id', 0)
                score = hit['_score']  # 获取相似度分数
                
                print(f"\n结果 {i}:")
                print(f"[TITLE] 标题: {title}")
                print(f"[PATH] 路径: {file_path}")
                print(f"[CHUNK_ID] 块ID: {chunk_id}")
                print(f"[SCORE] 相似度分数: {score:.4f}")
                
                if 'highlight' in hit:
                    highlight = hit['highlight']
                    if 'content' in highlight:
                        content_snippet = '...'.join(highlight['content'][:2])  # 显示前2个匹配片段
                        print(f"[MATCH] 匹配内容: {content_snippet[:300]}...")  # 限制显示长度
                else:
                    # 如果没有高亮，显示内容的前300个字符
                    content = source.get('content', '')
                    print(f"[CONTENT] 内容预览: {content[:300]}...")
                
                print("-" * 80)
        else:
            print("未找到匹配的文档。")
            
    except Exception as e:
        print(f"搜索过程中出现错误: {e}")

def main():
    """主函数"""
    try:
        # 连接到Elasticsearch并索引文档
        result = index_documents_to_es()
        if result is None:
            print("Elasticsearch连接失败，无法继续执行搜索")
            return
        
        es, index_name = result
        
        # 执行向量搜索
        search_query = "工伤保险和雇主险有什么区别？"
        print(f"\n[INFO] 正在使用向量搜索: {search_query}")
        search_in_documents(es, index_name, search_query)
        
        # 还可以尝试其他查询
        additional_queries = [
            "雇主责任险的保障范围",
            "平安企业团体综合意外险",
            "施工保的特点"
        ]
        
        print(f"\n[INFO] 尝试其他查询...")
        for query in additional_queries:
            print(f"\n{'='*60}")
            print(f"执行查询: {query}")
            print(f"{'='*60}")
            search_in_documents(es, index_name, query)
            
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("请确保Elasticsearch服务已在 https://localhost:9200 启动")
        print("并且DASHSCOPE_API_KEY环境变量已设置")


if __name__ == "__main__":
    main()