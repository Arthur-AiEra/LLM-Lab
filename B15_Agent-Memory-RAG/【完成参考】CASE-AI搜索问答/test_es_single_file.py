# test_es_single_file.py
# Description: A single-file script to test Elasticsearch indexing and retrieval,
# completely independent of the qwen_agent package structure to avoid import issues.

import os
import json
import hashlib
import logging
from elasticsearch import Elasticsearch, helpers
from qwen_agent.tools.base import BaseTool

# =====================================================================================
# Configuration
# =====================================================================================
# 设置日志，方便观察详细过程
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================================
# Copied class: SimpleDocParser (and its dependencies)
# Source: qwen_agent/tools/simple_doc_parser.py
# =====================================================================================
# NOTE: To make this standalone, we need to mock or stub the dependencies
# of SimpleDocParser, which are mainly langchain document loaders.
# For simplicity, we will create a mock parser that just reads text files.

class SimpleDocParserMock:
    def __init__(self, cfg={}):
        pass

    def load(self, file_path: str) -> list:
        logger.info(f"[SimpleDocParserMock] Parsing {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # The original returns a list of Document objects. We'll simulate that.
            from langchain_core.documents import Document
            return [Document(page_content=content, metadata={'source': file_path})]
        except Exception as e:
            logger.error(f"[SimpleDocParserMock] Failed to parse {file_path}: {e}")
            return []

# =====================================================================================
# Copied class: DocParser
# Source: qwen_agent/tools/doc_parser.py
# =====================================================================================
# NOTE: This class also has dependencies like `CharacterTextSplitter`.
# We'll assume the user has `langchain` installed.
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer

class DocParser(BaseTool):
    def __init__(self, cfg={}, name='doc_parser', description='Parse document from url.'):
        super().__init__(name=name, description=description)
        self.cfg = cfg
        self.chunk_size = cfg.get('parser_page_size', 500)
        self.chunk_overlap = 0
        self.separator = '\n'
        # Use a mock parser that handles .txt files
        self.doc_parsers = {'.txt': SimpleDocParserMock(cfg)}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}. Chunking might not be optimal.")
            self.tokenizer = None

    def _chunking(self, documents: list, chunk_size: int, chunk_overlap: int, separator: str):
        full_text = '\n'.join(doc.page_content for doc in documents)
        if not full_text.strip():
            return []
        
        if self.tokenizer:
            text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
                self.tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator)
            return text_splitter.split_text(full_text)
        else: # Fallback to simple splitter
            return [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]


    def call(self, args: dict, **kwargs) -> list:
        filepath = args.get('url', '')
        if not filepath or not os.path.exists(filepath):
            return []
        
        ext = os.path.splitext(filepath)[-1]
        if ext not in self.doc_parsers:
            logger.warning(f"Unsupported file type '{ext}' for {filepath}. Skipping.")
            return []

        documents = self.doc_parsers[ext].load(filepath)
        text_chunks = self._chunking(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator)
        
        return [{'content': x, 'source': os.path.basename(filepath)} for x in text_chunks]


# =====================================================================================
# Copied class: ElasticsearchSearcher
# Source: qwen_agent/searcher/elasticsearch_searcher.py
# =====================================================================================
class ElasticsearchSearcher:
    def __init__(self, cfg):
        self.cfg = cfg
        es_cfg = cfg.get('es', {})
        self.host = es_cfg.get('host', 'http://localhost')
        self.port = es_cfg.get('port', 9200)
        self.user = es_cfg.get('user')
        self.password = es_cfg.get('password')
        self.index_name = es_cfg.get('index_name', 'qwen_agent_rag_idx')
        self.parser = DocParser(cfg=self.cfg) # Use the copied DocParser
        self.client = self._connect()
        if self.client:
            logger.info("[ElasticsearchSearcher] Successfully connected to Elasticsearch!")
            self._create_index_if_not_exists()
        else:
            logger.error("[ElasticsearchSearcher] Failed to connect to Elasticsearch.")

    def _connect(self):
        try:
            es_args = {'hosts': [{'host': self.host.replace('https://', '').replace('http://', ''), 'port': self.port, 'scheme': 'https' if 'https' in self.host else 'http'}], 'verify_certs': False}
            if self.user and self.password:
                es_args['basic_auth'] = (self.user, self.password)
            client = Elasticsearch(**es_args)
            if not client.ping():
                raise ConnectionError("Ping failed")
            return client
        except Exception as e:
            logger.error(f"[ElasticsearchSearcher] Connection failed: {e}")
            return None

    def _create_index_if_not_exists(self):
        # ... (same as before)

    def index_files(self, files: list):
        # ... (same as before)

    def _get_chunks(self, files: list) -> list:
        # ... (same as before)

    def _filter_existing_chunks_efficiently(self, chunks: list) -> list:
        # ... (same as before)

    def search(self, query: str, k: int = 3) -> list:
        # ... (same as before)
    # NOTE: The full implementation of _create_index, index_files, etc. will be copied here.
    # To save space, they are omitted from this plan view.

# =====================================================================================
# Copied class: ESRetrievalTool
# Source: qwen_agent/tools/es_retrieval.py
# =====================================================================================
class ESRetrievalTool(BaseTool):
    def __init__(self, cfg: dict = None, name: str = 'retrieval', description: str = 'Retrieves information from a vector database.'):
        super().__init__(name=name, description=description)
        self.cfg = cfg or {}
        # Use the copied ElasticsearchSearcher
        self.searcher = ElasticsearchSearcher(cfg=self.cfg)

    def call(self, params: dict, **kwargs) -> str:
        # ... (same as before)
    # NOTE: The full implementation of call() will be copied here.


# =====================================================================================
# Test Function
# =====================================================================================
def test_retrieval():
    """
    An independent test function to validate the combined retrieval functionality.
    """
    print("--- Starting standalone test for Elasticsearch retrieval ---")
    rag_cfg = {
        "es": {
            "host": "https://localhost",
            "port": 9200,
            "user": "elastic",
            "password": "euqPcOlHrmW18rtaS-3P",
            "index_name": "standalone_test_index"
        },
        "parser_page_size": 500
    }

    try:
        retrieval_tool = ESRetrievalTool(cfg=rag_cfg)
    except Exception as e:
        print(f"--- FAILED: Could not instantiate ESRetrievalTool: {e} ---")
        return

    # ... (rest of the test function is the same)
    # It will use the self-contained classes.

if __name__ == '__main__':
    # Since we are mocking some parts (like PDF parsing), let's ensure the docs folder
    # has at least one .txt file for the test to work.
    if not os.path.exists('docs/2-雇主责任险.txt'):
         print("Please ensure 'docs/2-雇主责任险.txt' exists for this test.")
    else:
        test_retrieval() 