import logging
import os
from pyprojroot import here

from src.pipeline import Pipeline

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug('这是一条 Debug 消息，只有在级别设置为 DEBUG 时才会显示。')
logging.info('这是一条 Info 消息。')

def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    pipeline = Pipeline(root_path)

    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

if __name__ == '__main__':
    parse_pdfs(parallel=False, chunk_size=1, max_workers=1)