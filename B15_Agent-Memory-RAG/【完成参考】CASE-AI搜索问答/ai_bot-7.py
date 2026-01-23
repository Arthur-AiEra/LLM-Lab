# ai_bot-7.py
import os
import asyncio
from typing import Optional, Generator
from qwen_agent.agents import Assistant
import warnings
import gradio as gr
import time

warnings.filterwarnings("ignore")

def init_agent_service():
    """初始化具备 Elasticsearch RAG 和网络搜索能力的助手服务"""
    
    # ... (省略与 ai_bot-6.py 相同的代码) ...
    # 步骤 1: LLM 配置
    llm_cfg = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    # 步骤 2: RAG 配置
    rag_cfg = {
        "rag_backend": "elasticsearch",
        "es": {
            "host": "https://localhost",
            "port": 9200,
            "user": "elastic",
            "password": "euqPcOlHrmW18rtaS-3P",
            "index_name": "my_insurance_docs_index"
        },
        "parser_page_size": 500
    }

    # 步骤 3: 系统指令和工具
    system_instruction = '''你是一个AI助手。
请根据用户的问题，优先利用检索工具从本地知识库中查找最相关的信息。
如果本地知识库没有相关信息，再使用 tavily_search 工具从互联网上搜索，并结合这些信息给出专业、准确的回答。'''

    tools_cfg = [{
        "mcpServers": {
            "tavily-mcp": {
                "command": "npx",
                "args": ["-y", "tavily-mcp@0.1.4"],
                "env": {
                    "TAVILY_API_KEY": os.getenv('TAVILY_API_KEY', "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")
                },
                "disabled": False,
                "autoApprove": []
            }
        }
    }]

    # 获取文件夹下所有文件
    file_dir = os.path.join(os.path.dirname(__file__), 'docs')
    files = []
    if os.path.exists(file_dir):
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):
                files.append(file_path)
    print('知识库文件列表:', files)

    # 步骤 4: 创建智能体实例
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools_cfg,
        files=files,
        rag_cfg=rag_cfg
    )
    return bot

# 全局变量
bot = init_agent_service()
session_histories = {}

def get_session_id():
    return str(time.time())

def stream_predict(query: str, history: list, session_id: str) -> Generator:
    """Gradio 的核心预测函数 - 支持流式响应"""
    if session_id not in session_histories:
        session_histories[session_id] = []
    
    messages = session_histories[session_id]
    messages.append({'role': 'user', 'content': query})
    
    history[-1][1] = ""
    full_response = ""

    for response in bot.run(messages=messages):
        if response and response[-1]['role'] == 'assistant':
            new_text = response[-1]['content']
            if new_text != full_response:
                delta = new_text[len(full_response):]
                history[-1][1] += delta
                full_response = new_text
                yield history

    messages.append({'role': 'assistant', 'content': full_response})
    session_histories[session_id] = messages

def main():
    """启动自定义的 Gradio Web 图形界面"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir_path = os.path.join(current_dir, "static")
    
    # 使用 <link> 标签在 head 中加载 CSS
    app_head = f"""
    <head>
        <link rel="stylesheet" href="file=styles.css">
    </head>
    """
    
    with gr.Blocks(head=app_head, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")) as demo:
        session_id = gr.State(get_session_id)
        
        with gr.Row():
            with gr.Column(scale=2, elem_id="sidebar"):
                with gr.Row(elem_id="logo"):
                    # 使用 file= 格式加载 Logo
                    gr.HTML('<img id="logo-img" src="file=logo.png" alt="logo">')
                    gr.HTML('<h1 id="logo-text">知乎直答</h1>')
                
                gr.Button("🔍  搜索", elem_classes=["sidebar-btn", "active"])
                knowledge_btn = gr.Button("📚  知识库", elem_classes="sidebar-btn")
                favorites_btn = gr.Button("⭐  收藏", elem_classes="sidebar-btn")
                history_btn = gr.Button("🕒  历史", elem_classes="sidebar-btn")
            
            with gr.Column(scale=8, elem_id="main-chat"):
                # ... (省略与 ai_bot-6.py 相同的布局代码) ...
                with gr.Row(elem_id="chat-header"):
                    gr.HTML('<h1 id="chat-header-title">用提问发现世界</h1><p id="chat-header-subtitle">输入你的问题，或使用「@快捷引用」对知乎答主、知识库进行提问</p>')

                chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, height=550)
                
                with gr.Row(elem_id="suggestion-row") as suggestion_row:
                    suggestions = ['介绍下雇主责任险', '雇主责任险和工伤保险有什么主要区别？', '最近有什么新的保险产品推荐吗？']
                    suggestion_btns = []
                    for s in suggestions:
                        btn = gr.Button(s, elem_classes="suggestion-btn")
                        suggestion_btns.append(btn)
                
                with gr.Row(elem_id="input-container-wrapper"):
                    with gr.Row(elem_id="input-container"):
                        textbox = gr.Textbox(container=False, show_label=False, placeholder="输入你的问题...", scale=10)
                        submit_btn = gr.Button("↑", scale=1, min_width=0, variant="primary")
        
        def on_submit(query, history):
            history.append([query, None])
            return "", history

        def on_suggestion_click(suggestion, history):
            history.append([suggestion, None])
            return "", history, gr.update(visible=False)
        
        def show_not_implemented_toast():
            gr.Info("功能暂未实现，敬请期待！")

        knowledge_btn.click(show_not_implemented_toast, None, None)
        favorites_btn.click(show_not_implemented_toast, None, None)
        history_btn.click(show_not_implemented_toast, None, None)

        submit_event = textbox.submit(on_submit, [textbox, chatbot], [textbox, chatbot], queue=False)
        submit_event.then(lambda: gr.update(visible=False), None, suggestion_row)
        submit_event.then(stream_predict, [textbox, chatbot, session_id], chatbot)

        click_event = submit_btn.click(on_submit, [textbox, chatbot], [textbox, chatbot], queue=False)
        click_event.then(lambda: gr.update(visible=False), None, suggestion_row)
        click_event.then(stream_predict, [textbox, chatbot, session_id], chatbot)
        
        for btn in suggestion_btns:
            s_click_event = btn.click(on_suggestion_click, [btn, chatbot], [textbox, chatbot, suggestion_row], queue=False)
            s_click_event.then(stream_predict, [btn, chatbot, session_id], chatbot)

    print("正在启动 AI 助手 Web 界面 (v7)...")
    demo.launch(allowed_paths=[static_dir_path])

if __name__ == '__main__':
    main() 