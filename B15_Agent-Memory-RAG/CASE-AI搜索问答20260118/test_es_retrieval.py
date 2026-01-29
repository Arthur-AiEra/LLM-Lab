import os
from qwen_agent.tools.es_retrieval import ESRetrievalTool

# 测试ES检索工具
def test_es_retrieval():
    # 创建ES检索工具实例
    cfg = {
        'es_host': 'localhost',
        'es_port': 9200,
        'es_username': 'elastic',
        'es_password': '5A7C1+=PbQCpkw1jvu-8',
        'index_name': 'test_docs'
    }
    
    try:
        tool = ESRetrievalTool(cfg)
        print("ES检索工具初始化成功!")
        
        # 测试参数
        params = {
            'query': '雇主责任险',
            'files': ['./docs/2-雇主责任险.txt']  # 使用存在的文档文件
        }
        
        # 执行检索
        results = tool.call(params)
        print(f"检索结果: {results}")
        
    except Exception as e:
        print(f"ES检索工具测试失败: {e}")

if __name__ == "__main__":
    test_es_retrieval()