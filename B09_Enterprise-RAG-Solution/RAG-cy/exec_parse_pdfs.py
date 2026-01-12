import logging
import os
from pyprojroot import here

from src.pipeline import Pipeline

## 解决 docling 解析pdf 异常
# 1. 清理Hugging Face 的缓存目录 rm -rf ~/.cache/huggingface/hub
# 2. pip install --upgrade certifi
# 3. 禁用 OpenMP 并行化（最可能有效）： 崩溃发生在 PyTorch 的并行处理部分。通过强制 PyTorch 在单线程中运行，通常可以避免这个 bug
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug('这是一条 Debug 消息，只有在级别设置为 DEBUG 时才会显示。')
logging.info('这是一条 Info 消息。')

def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    pipeline = Pipeline(root_path)

    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

if __name__ == '__main__':
    parse_pdfs(parallel=False, chunk_size=1, max_workers=3)