import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

# ====== 保险智能问答系统 prompt 和函数描述 ======
system_prompt = """我是保险智能问答助手，我可以解答关于各种保险产品的问题。我能处理以下类型的查询：
- 保险产品特性：保障范围、保险条款、理赔流程等
- 保险产品对比：不同保险产品的差异和适用场景
- 文档检索：从保险文档中查找相关信息并进行总结

以下是保险产品文档的相关信息：
- 平安商业综合责任保险（亚马逊）：适用于电商平台商家的责任保障
- 雇主责任险：企业对员工在工作期间发生意外的责任保障
- 平安企业团体综合意外险：为企业员工提供意外伤害保障
- 雇主安心保：专门为雇主设计的综合保障方案
- 施工保：针对建筑施工行业的专项保险
- 财产一切险：对企业财产进行全面保障的保险产品

每当 exc_sql 工具返回 markdown 表格和图片时，你必须原样输出工具返回的全部内容（包括图片 markdown），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。
"""

functions_desc = [
    {
        "name": "exc_sql",
        "description": "对于生成的SQL，进行SQL查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                }
            },
            "required": ["sql_input"],
        },
    },
]

# ====== 会话隔离 DataFrame 存储 ======
# 用于存储每个会话的 DataFrame，避免多用户数据串扰
_last_df_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id，这里用 messages 的 id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL查询工具，执行传入的SQL语句并返回结果，并自动进行可视化。
    """
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        args = json.loads(params)
        sql_input = args['sql_input']
        database = args.get('database', 'ubr')
        engine = create_engine(
            f'mysql+mysqlconnector://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/{database}?charset=utf8mb4',
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        try:
            df = pd.read_sql(sql_input, engine)
            md = df.head(10).to_markdown(index=False)
            # 自动创建目录
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'bar_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            # 生成图表
            generate_chart_png(df, save_path)
            img_path = os.path.join('image_show', filename)
            img_md = f'![柱状图]({img_path})'
            return f"{md}\n\n{img_md}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"

# ========== 通用可视化函数 ========== 
def generate_chart_png(df_sql, save_path):
    columns = df_sql.columns
    x = np.arange(len(df_sql))
    # 获取object类型
    object_columns = df_sql.select_dtypes(include='O').columns.tolist()
    if columns[0] in object_columns:
        object_columns.remove(columns[0])
    num_columns = df_sql.select_dtypes(exclude='O').columns.tolist()
    if len(object_columns) > 0:
        # 对数据进行透视，以便为每个日期和销售渠道创建堆积柱状图
        pivot_df = df_sql.pivot_table(index=columns[0], columns=object_columns, 
                                      values=num_columns, 
                                      fill_value=0)
        # 绘制堆积柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        # 为每个销售渠道和票类型创建柱状图
        bottoms = None
        for col in pivot_df.columns:
            ax.bar(pivot_df.index, pivot_df[col], bottom=bottoms, label=str(col))
            if bottoms is None:
                bottoms = pivot_df[col].copy()
            else:
                bottoms += pivot_df[col]
    else:
        print('进入到else...')
        bottom = np.zeros(len(df_sql))
        for column in columns[1:]:
            plt.bar(x, df_sql[column], bottom=bottom, label=column)
            bottom += df_sql[column]
        plt.xticks(x, df_sql[columns[0]])
    plt.legend()
    plt.title("销售统计")
    plt.xlabel(columns[0])
    plt.ylabel("门票数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ====== 初始化保险问答助手服务 ======
def init_agent_service():
    """初始化保险问答助手服务"""
    # 获取docs目录下所有文档文件
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, file)
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                files.append(file_path)

    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    
    # 配置使用ES向量检索
    rag_cfg = {
        'use_es_vector': True,  # 启用ES向量检索
        'es_host': 'localhost',
        'es_port': 9200,
        'es_username': 'elastic',
        'es_password': '5A7C1+=PbQCpkw1jvu-8',
        'vector_index_name': 'insurance_docs_vectors'
    }
    function_list = ['exc_sql',
                {
                    "mcpServers": {
                        "tavily-mcp": {
                            "command": "npx",
                            "args": [
                                "-y",
                                "tavily-mcp@0.1.4"
                            ],
                            "autoApprove": [],
                            "disabled": False,
                            "env": {
                                "TAVILY_API_KEY": os.getenv('TAVILY_API_KEY', 'your-api-key-here')
                            }
                        }
                    }
                }
        ]
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='保险智能问答助手',
            description='保险产品咨询与文档检索（向量搜索）',
            system_message=system_prompt,
            function_list=function_list,  # 移除绘图工具
            files=files,  # 添加文档文件
            rag_cfg=rag_cfg  # 添加RAG配置
        )
        print("保险问答助手初始化成功！（使用ES向量检索）")
        return bot
    except Exception as e:
        print(f"保险问答助手初始化失败 (ES向量连接问题): {str(e)}")
        print("尝试使用ES常规检索...")
        # 如果向量检索失败，尝试使用ES常规检索
        try:
            rag_cfg_fallback = {
                'use_es': True,
                'es_host': 'localhost',
                'es_port': 9200,
                'es_username': 'elastic',
                'es_password': '5A7C1+=PbQCpkw1jvu-8',
                'index_name': 'insurance_docs'
            }

            bot = Assistant(
                llm=llm_cfg,
                name='保险智能问答助手',
                description='保险产品咨询与文档检索（ES检索）',
                system_message=system_prompt,
                function_list=function_list,  # 移除绘图工具
                files=files,  # 添加文档文件
                rag_cfg=rag_cfg_fallback  # 添加RAG配置
            )
            print("保险问答助手初始化成功！（使用ES检索）")
            return bot
        except Exception as fallback_error:
            print(f"ES检索也失败: {str(fallback_error)}")
            print("尝试使用默认检索配置...")
            # 最终降级到默认配置
            try:
                bot = Assistant(
                    llm=llm_cfg,
                    name='保险智能问答助手',
                    description='保险产品咨询与文档检索（默认检索）',
                    system_message=system_prompt,
                    function_list=['exc_sql'],  # 移除绘图工具
                    files=files  # 添加文档文件
                )
                print("保险问答助手初始化成功！（使用默认检索）")
                return bot
            except Exception as final_error:
                print(f"备用初始化也失败: {str(final_error)}")
                raise

def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型保险查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '请介绍下雇主责任险的保障范围',
                '平安商业综合责任保险的理赔流程是什么？',
                '对比一下雇主安心保和普通雇主责任险的区别'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）