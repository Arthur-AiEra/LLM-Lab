import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pathlib import Path
import fitz  # PyMuPDF for PDF files
import docx  # python-docx for DOCX files
import json

# Elasticsearch连接配置
ES_HOST = 'localhost'
ES_PORT = 9200
ES_USERNAME = 'elastic'
ES_PASSWORD = ''

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

def index_documents_to_es():
    """连接到Elasticsearch，创建索引，索引文档"""
    # 连接到Elasticsearch
    es_url = f"http://{ES_HOST}:{ES_PORT}"
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
    index_name = 'insurance_documents'
    
    # 删除已存在的索引（如果存在）
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"已删除现有索引: {index_name}")
    
    # 创建新的索引 - 使用更简单的字段结构
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
                }
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, body=mapping)
        print(f"[SUCCESS] 成功创建索引 {index_name}")
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
    indexed_files = 0
    
    for file_path in docs_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
            print(f"[INFO] 处理文件: {file_path.name}")
            content = get_document_content(file_path)
            total_files += 1
            
            if content.strip():  # 确保文档内容不为空
                doc = {
                    "_index": index_name,
                    "_source": {
                        "title": file_path.name,
                        "content": content[:32000],  # 限制内容长度，避免超过ES限制
                        "file_path": str(file_path.absolute())
                    }
                }
                
                try:
                    es.index(index=index_name, body=doc["_source"])
                    print(f"[SUCCESS] 已索引文件: {file_path.name}")
                    indexed_files += 1
                except Exception as e:
                    print(f"[ERROR] 索引文件时出错 {file_path.name}: {e}")
            else:
                print(f"[WARN] 跳过空文档: {file_path.name}")
    
    # 刷新索引以使文档立即可用
    es.indices.refresh(index=index_name)
    print(f"[SUCCESS] 总共处理了 {total_files} 个文件，成功索引 {indexed_files} 个文档")
    
    return es, index_name

def clean_text_for_display(text):
    """清理文本以避免编码问题"""
    if not isinstance(text, str):
        return str(text)
    # 替换可能导致编码问题的字符
    replacements = {
        '\u2022': '*',  # bullet point
        '\u201c': '"',  # left double quotation mark
        '\u201d': '"',  # right double quotation mark
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def search_in_documents(es, index_name, search_query):
    """在Elasticsearch中搜索文档"""
    # 构建搜索查询，参考es_doc_search.py的查询方式
    query_body = {
        "query": {
            "multi_match": {
                "query": search_query,
                "fields": ["title", "content"],
                "type": "best_fields"
            }
        },
        "highlight": {
            "fields": {
                "content": {},
                "title": {}
            }
        },
        "size": 10
    }
    
    try:
        # 执行搜索
        response = es.search(index=index_name, body=query_body)
        
        # 显示搜索结果，参考es_doc_search.py的显示方式
        print(f"\n搜索查询: {search_query}")
        print(f"找到 {response['hits']['total']['value']} 个结果:\n")
        
        if response['hits']['total']['value'] > 0:
            print("="*60)
            
            for i, hit in enumerate(response['hits']['hits'], 1):
                source = hit['_source']
                title = source.get('title', '无标题')
                file_path = source.get('file_path', '未知路径')
                score = hit['_score']  # 获取相关性分数
                
                print(f"\n结果 {i}:")
                print(f"[TITLE] 标题: {clean_text_for_display(title)}")
                print(f"[PATH] 路径: {clean_text_for_display(file_path)}")
                print(f"[SCORE] 相关性分数: {score}")
                
                if 'highlight' in hit:
                    highlight = hit['highlight']
                    if 'content' in highlight:
                        content_snippet = '...'.join(highlight['content'][:3])  # 显示前3个匹配片段
                        clean_content = clean_text_for_display(content_snippet[:200])
                        print(f"[MATCH] 匹配内容: {clean_content}...")  # 限制显示长度
                    elif 'title' in highlight:
                        title_snippet = '...'.join(highlight['title'])
                        clean_title = clean_text_for_display(title_snippet)
                        print(f"[MATCH] 匹配标题: {clean_title}")
                else:
                    # 如果没有高亮，显示内容的前200个字符
                    content = source.get('content', '')
                    clean_content = clean_text_for_display(content[:200])
                    print(f"[PREVIEW] 内容预览: {clean_content}...")
                
                print("-" * 60)
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
        
        # 执行搜索 - 与es_doc_search.py相同的查询
        search_query = "工伤保险和雇主险有什么区别？"
        print(f"\n[INFO] 正在搜索: {search_query}")
        search_in_documents(es, index_name, search_query)
            
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("请确保Elasticsearch服务已在 https://localhost:9200 启动")


if __name__ == "__main__":
    main()