import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from langchain_core.tools import tool  # 导入 LangChain 的 tool 装饰器

import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# 创建 MCP Server
mcp = FastMCP("桌面 TXT 文件统计器")


# 注意：装饰器的顺序很重要。
# 我们先打 @tool (LangChain)，再打 @mcp.tool() (MCP)
# 或者反过来，通常建议将 @tool 放在最外层，这样导出的直接就是 LangChain 工具对象

@tool
@mcp.tool()
def count_desktop_txt_files() -> int:
    """统计桌面上 .txt 文件的数量"""
    try:
        # ... (这里也加上 try...except) ...
        _path = os.path.join(os.path.expanduser("~"), "Desktop")
        desktop_path = Path(_path)
        if not os.path.isdir(_path):
            return 0

        # 检查是否有读取权限
        if not os.access(_path, os.R_OK):
            logging.error("⚠️ 权限受限：macOS 拒绝访问桌面。")
            return -1

        txt_files = list(desktop_path.glob("*.txt"))
        return len(txt_files)
    except Exception as e:
        logging.error(f"执行 count_desktop_txt_files 时发生未知错误: {e}", exc_info=True)
        return -1  # 或者返回一个能表示错误的值
@tool
@mcp.tool()
def list_desktop_txt_files() -> str:
    """获取桌面上所有 .txt 文件的列表"""
    try:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

        # 检查路径是否存在
        if not os.path.isdir(desktop_path):
            logging.warning(f"桌面路径不存在: {desktop_path}")
            return f"错误：找不到桌面路径 '{desktop_path}'。"

        files = [f for f in os.listdir(desktop_path) if f.endswith(".txt")]

        if not files:
            return "信息：桌面上没有找到 .txt 文件。"

        return "桌面上找到的 .txt 文件有：\n" + "\n".join(files)

    except Exception as e:
        # 捕获所有可能的异常
        logging.error(f"执行 list_desktop_txt_files 时发生未知错误: {e}", exc_info=True)
        # 向 Agent 返回一个清晰的错误信息，而不是让服务崩溃
        return f"错误：在尝试列出桌面文件时发生服务器内部错误: {str(e)}"
@tool
@mcp.tool()
def read_txt_file(filename: str) -> str:
    """读取指定txt文件的内容

    Args:
        filename: txt文件的名称（例如：test.txt）
    """
    desktop_path = Path(os.path.expanduser("~/Desktop"))
    file_path = desktop_path / filename

    if not file_path.exists():
        logging.error(f"错误：文件 '{filename}' 不存在于桌面上。")
        return f"错误：文件 '{filename}' 不存在于桌面上。"

    if file_path.suffix.lower() != '.txt':
        logging.error(f"错误：文件 '{filename}' 不是txt文件。")
        return f"错误：文件 '{filename}' 不是txt文件。"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logging.info(f"文件 '{filename}' 的内容：\n\n{content}")
        return f"文件 '{filename}' 的内容：\n\n{content}"
    except Exception as e:
        logging.error(f"读取文件时发生错误：{str(e)}")
        return f"读取文件时发生错误：{str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行桌面 TXT 文件统计器 MCP Server")
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="stdio",
        help="选择传输协议 (sse 或 stdio，默认为 stdio)"
    )
    args = parser.parse_args()

    logging.info(f"正在以 {args.transport} 模式启动 MCP Server...")

    _port = 8000
    if args.transport == "sse":
        _port = 8001

    mcp.settings.port = _port
    mcp.run(transport=args.transport)
