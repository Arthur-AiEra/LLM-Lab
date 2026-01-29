import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import gradio as gr
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np
from datetime import datetime


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

    function_list = ['exc_sql']

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
                    function_list=function_list,  # 移除绘图工具
                    files=files  # 添加文档文件
                )
                print("保险问答助手初始化成功！（使用默认检索）")
                return bot
            except Exception as final_error:
                print(f"备用初始化也失败: {str(final_error)}")
                raise


def app_custom_gradio():
    """自定义Gradio界面，参考知乎直答设计"""
    bot = init_agent_service()
    
    with gr.Blocks(
        title="保险智能问答助手",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            spacing_size="sm",
            radius_size="sm"
        ),
        css="""
        .container { max-width: 100% !important; }
        #logo { 
            text-align: center; 
            padding: 10px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-bottom: 1px solid #e0e0e0;
        }
        .logo-text { 
            font-size: 20px; 
            font-weight: bold; 
            color: #6C5CE7;
            margin: 0;
        }
        .hero-title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin: 20px 0;
            color: #2D3436;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 16px;
            color: #636e72;
            margin-bottom: 30px;
        }
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .suggestion-cards {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 30px;
        }
        .suggestion-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .suggestion-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #6C5CE7;
        }
        .chat-container {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #d1d8ff;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: left;
        }
        .bot-message {
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: left;
            border: 1px solid #e0e0e0;
        }
        """
    ) as demo:
        
        # 侧边栏
        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                with gr.Group(elem_id="logo"):
                    gr.Markdown("### <span style='color: #6C5CE7;'>🛡️ 保险智能助手</span>", elem_id="logo")
                
                with gr.Accordion("🔍 知识库", open=True):
                    gr.Markdown("已加载保险文档:")
                    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
                    if os.path.exists(docs_dir):
                        docs_list = os.listdir(docs_dir)[:10]  # 显示前10个文档
                        for doc_file in docs_list:
                            gr.Markdown(f"- {doc_file}")
                    
                with gr.Accordion("⚙️ 设置", open=False):
                    deep_think = gr.Checkbox(label="深度思考模式", value=False, interactive=True)
                    
            # 主内容区
            with gr.Column(scale=4):
                # 标题区域
                gr.Markdown(
                    "<h1 class='hero-title'>用提问<span style='color: #0984E3;'>发现</span>保险世界</h1>",
                    elem_classes=["hero-title"]
                )
                gr.Markdown(
                    "专业解答各类保险问题，保障范围、理赔流程、产品对比等",
                    elem_classes=["hero-subtitle"]
                )
                
                # 聊天历史显示
                chat_history = gr.Chatbot(
                    label="对话历史",
                    elem_classes=["chat-container"],
                    bubble_full_width=False,
                    avatar_images=(
                        "https://img.icons8.com/color/96/000000/user--v1.png",  # 用户头像
                        "https://img.icons8.com/fluency/96/000000/customer-service.png"  # 助手头像
                    )
                )
                
                # 输入区域
                with gr.Row(elem_classes=["input-container"]):
                    user_input = gr.Textbox(
                        placeholder="输入你的问题，或使用「@快捷引用」对保险文档进行提问...",
                        lines=2,
                        max_lines=5,
                        elem_id="user-input"
                    )
                    submit_btn = gr.Button("🚀 提问", variant="primary")
                    attach_btn = gr.Button("📎 附件")
                
                # 功能开关
                with gr.Row():
                    deep_think_toggle = gr.Checkbox(label="启用深度思考", value=False)
                    knowledge_source = gr.Dropdown(
                        choices=["保险文档库", "公开知识", "SQL数据库"],
                        value="保险文档库",
                        label="知识源"
                    )
                
                # 推荐问题
                gr.Markdown("### 💡 推荐问题", elem_id="suggestions-header")
                with gr.Row(elem_classes=["suggestion-cards"]):
                    with gr.Column():
                        suggestion1 = gr.Button("请介绍下雇主责任险的保障范围", variant="secondary")
                        suggestion2 = gr.Button("平安商业综合责任保险的理赔流程是什么？", variant="secondary")
                    with gr.Column():
                        suggestion3 = gr.Button("对比一下雇主安心保和普通雇主责任险的区别", variant="secondary")
                        suggestion4 = gr.Button("财产一切险的适用场景有哪些？", variant="secondary")
                
                # 事件处理
                def respond(message, chat_history, deep_think_flag):
                    if not message:
                        return "", chat_history
                    
                    # 添加用户消息到历史
                    updated_chat_history = chat_history + [(message, None)]
                    
                    # 构建消息列表
                    messages = [{'role': 'user', 'content': message}]
                    
                    # 运行助手获取响应
                    response = []
                    for response_piece in bot.run(messages):
                        response = response_piece
                    
                    # 获取助手的回复内容
                    if response and len(response) > 0:
                        bot_response = response[0].get('content', '抱歉，我没有理解您的问题。')
                    else:
                        bot_response = "抱歉，暂时无法回答您的问题。"
                    
                    # 更新聊天历史
                    updated_chat_history[-1] = (message, bot_response)
                    
                    return "", updated_chat_history
                
                # 绑定提交按钮事件
                submit_btn.click(
                    respond,
                    inputs=[user_input, chat_history, deep_think_toggle],
                    outputs=[user_input, chat_history]
                )
                
                # 绑定回车事件
                user_input.submit(
                    respond,
                    inputs=[user_input, chat_history, deep_think_toggle],
                    outputs=[user_input, chat_history]
                )
                
                # 绑定推荐问题按钮
                for btn in [suggestion1, suggestion2, suggestion3, suggestion4]:
                    btn.click(
                        lambda x=btn.value: x,
                        outputs=user_input
                    )

    # 启动界面
    demo.launch(share=False, debug=True, show_error=True)


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
    # 启动自定义Gradio界面
    app_custom_gradio()