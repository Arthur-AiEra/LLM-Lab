## Python 代码逐行解析：`assistant_ticket_bot-1.py`

本文档将对 `assistant_ticket_bot-1.py` 文件中的每一行代码进行详细的中文解析，解释其功能和作用。

```python
1	"""
2	门票助手 - Streamlit Web 界面版本
3	使用 OpenAI SDK 和 Streamlit 构建交互式 Web 应用
4	"""
```
**解析:**
*   **第 1-4 行:** 这是一个多行字符串，作为文件的文档字符串（docstring）。它简要描述了该 Python 文件的目的：一个基于 Streamlit Web 界面和 OpenAI SDK 构建的交互式门票助手应用。

```python
6	import os
7	import json
8	import pandas as pd
9	import streamlit as st
10	from openai import OpenAI, APIError
11	from sqlalchemy import create_engine
```
**解析:**
*   **第 6 行:** `import os` 导入 `os` 模块，提供了与操作系统交互的功能，例如访问环境变量。
*   **第 7 行:** `import json` 导入 `json` 模块，用于处理 JSON (JavaScript Object Notation) 数据格式，常用于数据序列化和反序列化。
*   **第 8 行:** `import pandas as pd` 导入 `pandas` 库并将其别名为 `pd`。`pandas` 是一个强大的数据处理和分析库，主要用于处理表格数据（DataFrame）。
*   **第 9 行:** `import streamlit as st` 导入 `streamlit` 库并将其别名为 `st`。`Streamlit` 是一个用于快速构建数据应用和机器学习工具的 Python 库，它允许开发者用纯 Python 代码创建交互式 Web 界面。
*   **第 10 行:** `from openai import OpenAI, APIError` 从 `openai` 库中导入 `OpenAI` 类和 `APIError` 异常。`OpenAI` 类用于与 OpenAI API 进行交互，`APIError` 用于捕获 API 调用过程中可能发生的错误。
*   **第 11 行:** `from sqlalchemy import create_engine` 从 `sqlalchemy` 库中导入 `create_engine` 函数。`SQLAlchemy` 是一个 Python SQL 工具包和对象关系映射 (ORM) 库，`create_engine` 用于创建数据库连接引擎。

```python
14	st.set_page_config(
15	    page_title="门票助手",
16	    page_icon="🎫",
17	    layout="wide",
18	    initial_sidebar_state="expanded"
19	)
```
**解析:**
*   **第 14-19 行:** `st.set_page_config()` 是 Streamlit 的一个函数，用于配置 Web 应用的页面设置。
    *   `page_title="门票助手"`: 设置浏览器标签页的标题为“门票助手”。
    *   `page_icon="🎫"`: 设置浏览器标签页的图标为一个票据表情符号。
    *   `layout="wide"`: 设置页面布局为宽屏模式，使内容可以占据更宽的屏幕空间。
    *   `initial_sidebar_state="expanded"`: 设置侧边栏的初始状态为展开。

```python
22	SYSTEM_PROMPT = """我是门票助手，以下是关于门票订单表相关的字段，我可能会编写对应的SQL(运行环境是MySQL8.0)，对数据进行查询
23	-- 门票订单表
24	CREATE TABLE tkt_orders (
25	    order_time DATETIME,             -- 订单日期
26	    account_id INT,                  -- 预定用户ID
27	    gov_id VARCHAR(18),              -- 商品使用人ID（身份证号）
28	    gender VARCHAR(10),              -- 使用人性别
29	    age INT,                         -- 年龄
30	    province VARCHAR(30),           -- 使用人省份
31	    SKU VARCHAR(100),                -- 商品SKU名
32	    product_serial_no VARCHAR(30),  -- 商品ID
33	    eco_main_order_id VARCHAR(20),  -- 订单ID
34	    sales_channel VARCHAR(20),      -- 销售渠道
35	    status VARCHAR(30),             -- 商品状态
36	    order_value DECIMAL(10,2),       -- 订单金额
37	    quantity INT                     -- 商品数量
38	);
39	一日门票，对应多种SKU：
40	Universal Studios Beijing One-Day Dated Ticket-Standard
41	Universal Studios Beijing One-Day Dated Ticket-Child
42	Universal Studios Beijing One-Day Dated Ticket-Senior
43	二日门票，对应多种SKU：
44	USB 1.5-Day Dated Ticket Standard
45	USB 1.5-Day Dated Ticket Discounted
46	一日门票、二日门票查询
47	SUM(CASE WHEN SKU LIKE 'Universal Studios Beijing One-Day%' THEN quantity ELSE 0 END) AS one_day_ticket_sales,
48	SUM(CASE WHEN SKU LIKE 'USB%' THEN quantity ELSE 0 END) AS two_day_ticket_sales
49	我将回答用户关于门票相关的问题
50	"""
```
**解析:**
*   **第 22-50 行:** 定义了一个名为 `SYSTEM_PROMPT` 的多行字符串变量。这是一个系统提示词，用于指导 OpenAI 模型在对话中的行为和知识背景。
    *   它首先声明自己是“门票助手”，并说明可能会编写 SQL 查询来查询门票订单表。
    *   接着，它提供了一个 `tkt_orders` 表的 `CREATE TABLE` 语句，详细列出了表的结构和每个字段的含义（如 `order_time`、`account_id`、`SKU` 等）。这为模型提供了数据库模式信息。
    *   然后，它列举了两种类型的门票（一日门票和二日门票）及其对应的 SKU 示例，帮助模型理解不同门票的标识。
    *   最后，它提供了一个 SQL 查询示例，用于统计一日门票和二日门票的销量，这为模型提供了如何进行数据聚合的参考。
    *   整个提示词旨在让模型理解门票业务逻辑和数据库结构，以便能准确地回答用户关于门票的问题并生成正确的 SQL 查询。

```python
53	TOOLS = [
54	    {
55	        "type": "function",
56	        "function": {
57	            "name": "execute_sql",
58	            "description": "执行SQL查询语句，用于查询门票订单数据",
59	            "parameters": {
60	                "type": "object",
61	                "properties": {
62	                    "sql_query": {
63	                        "type": "string",
64	                        "description": "要执行的SQL查询语句"
65	                    },
66	                    "database": {
67	                        "type": "string",
68	                        "description": "数据库名称，默认为 'ubr'",
69	                        "default": "ubr"
70	                    }
71	                },
72	                "required": ["sql_query"]
73	            }
74	        }
75	    }
76	]
```
**解析:**
*   **第 53-76 行:** 定义了一个名为 `TOOLS` 的列表，其中包含一个字典，用于描述 OpenAI 模型可以调用的工具。这是 OpenAI Function Calling 功能的配置。
    *   `"type": "function"`: 表明这是一个函数工具。
    *   `"function"`: 包含函数工具的具体定义。
        *   `"name": "execute_sql"`: 定义了工具的名称，模型在需要执行 SQL 查询时会调用这个名称的函数。
        *   `"description": "执行SQL查询语句，用于查询门票订单数据"`: 描述了工具的功能，这有助于模型理解何时以及如何使用该工具。
        *   `"parameters"`: 定义了函数工具的参数。
            *   `"type": "object"`: 参数是一个 JSON 对象。
            *   `"properties"`: 定义了对象的属性（即函数的参数）。
                *   `"sql_query"`: 类型为字符串，描述为“要执行的SQL查询语句”，这是必需参数。
                *   `"database"`: 类型为字符串，描述为“数据库名称，默认为 'ubr'”，这是一个可选参数，默认值为 'ubr'。
            *   `"required": ["sql_query"]`: 指定 `sql_query` 是调用此工具时必须提供的参数。

```python
79	DB_CONFIG = {
80	    'host': 'rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com',
81	    'port': 3306,
82	    'user': 'student123',
83	    'password': 'student321',
84	    'charset': 'utf8mb4'
85	}
```
**解析:**
*   **第 79-85 行:** 定义了一个名为 `DB_CONFIG` 的字典，用于存储数据库连接的配置信息。
    *   `'host'`: 数据库服务器的主机地址。
    *   `'port'`: 数据库服务器的端口号。
    *   `'user'`: 连接数据库的用户名。
    *   `'password'`: 连接数据库的密码。
    *   `'charset'`: 数据库连接使用的字符集，这里是 `utf8mb4`，支持更广泛的字符。

```python
88	def execute_sql(sql_query: str, database: str = 'ubr') -> str:
89	    """执行 SQL 查询并返回结果"""
90	    try:
91	        connection_string = (
92	            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
93	            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{database}"
94	            f"?charset={DB_CONFIG['charset']}"
95	        )
96	
97	        engine = create_engine(
98	            connection_string,
99	            connect_args={'connect_timeout': 10},
100	            pool_size=10,
101	            max_overflow=20
102	        )
103	
104	        df = pd.read_sql(sql_query, engine)
105	        result = df.head(10).to_markdown(index=False)
106	        return result if result else "查询结果为空"
107	
108	    except Exception as e:
109	        return f"SQL 执行出错: {str(e)}"
```
**解析:**
*   **第 88 行:** 定义了一个名为 `execute_sql` 的函数，它接受两个参数：`sql_query` (字符串类型，表示要执行的 SQL 查询语句) 和 `database` (字符串类型，默认为 'ubr'，表示数据库名称)。函数返回一个字符串类型的结果。
*   **第 89 行:** 函数的文档字符串，简要说明函数的功能是“执行 SQL 查询并返回结果”。
*   **第 90 行:** `try:` 语句块开始，用于捕获可能发生的异常。
*   **第 91-95 行:** 构建数据库连接字符串 `connection_string`。它使用 f-string 格式化，将 `DB_CONFIG` 字典中的数据库连接信息（用户、密码、主机、端口、数据库名和字符集）组合成一个符合 SQLAlchemy 规范的连接字符串。
*   **第 97-102 行:** 使用 `create_engine()` 函数创建一个 SQLAlchemy 数据库引擎。
    *   `connection_string`: 上一步构建的连接字符串。
    *   `connect_args={'connect_timeout': 10}`: 设置连接超时时间为 10 秒。
    *   `pool_size=10`: 设置连接池中保持的连接数。
    *   `max_overflow=20`: 设置连接池允许的最大溢出连接数。
*   **第 104 行:** `pd.read_sql(sql_query, engine)` 使用 pandas 的 `read_sql` 函数执行 SQL 查询，并将结果读取到一个 DataFrame `df` 中。
*   **第 105 行:** `df.head(10).to_markdown(index=False)` 将 DataFrame 的前 10 行转换为 Markdown 格式的字符串，并且不包含索引。`result` 变量存储这个 Markdown 字符串。
*   **第 106 行:** `return result if result else "查询结果为空"` 如果 `result` 不为空，则返回 `result`；否则返回字符串“查询结果为空”。
*   **第 108 行:** `except Exception as e:` 捕获所有类型的异常，并将其赋值给变量 `e`。
*   **第 109 行:** `return f"SQL 执行出错: {str(e)}"` 如果发生异常，则返回一个包含错误信息的字符串。

```python
112	def process_tool_call(tool_name: str, tool_input: dict) -> str:
113	    """处理 OpenAI 返回的工具调用"""
114	    if tool_name == "execute_sql":
115	        return execute_sql(
116	            sql_query=tool_input.get('sql_query'),
117	            database=tool_input.get('database', 'ubr')
118	        )
119	    else:
120	        return f"未知的工具: {tool_name}"
```
**解析:**
*   **第 112 行:** 定义了一个名为 `process_tool_call` 的函数，用于处理 OpenAI 模型返回的工具调用。它接受 `tool_name` (工具名称，字符串类型) 和 `tool_input` (工具参数，字典类型) 作为参数，并返回一个字符串。
*   **第 113 行:** 函数的文档字符串，说明其功能是“处理 OpenAI 返回的工具调用”。
*   **第 114 行:** `if tool_name == "execute_sql":` 检查工具名称是否为“execute_sql”。
*   **第 115-118 行:** 如果工具名称是“execute_sql”，则调用前面定义的 `execute_sql` 函数，并从 `tool_input` 字典中获取 `sql_query` 和 `database` 参数。`database` 参数如果不存在，则默认为 'ubr'。
*   **第 119 行:** `else:` 如果工具名称不是“execute_sql”。
*   **第 120 行:** `return f"未知的工具: {tool_name}"` 返回一个表示工具未知的错误信息。

```python
123	if "messages" not in st.session_state:
124	    st.session_state.messages = []
125	
126	if "client" not in st.session_state:
127	    api_key = os.getenv('OPENAI_API_KEY')
128	    if not api_key:
129	        st.error("❌ OPENAI_API_KEY 环境变量未设置")
130	        st.stop()
131	    st.session_state.client = OpenAI(api_key=api_key, base_url="https://api.fe8.cn/v1")
```
**解析:**
*   **第 123-124 行:** 检查 Streamlit 的 `session_state` 中是否存在 `"messages"` 键。如果不存在，则初始化 `st.session_state.messages` 为一个空列表。`session_state` 用于在 Streamlit 应用的不同重新运行之间持久化数据，`messages` 列表将用于存储聊天历史。
*   **第 126-131 行:** 检查 `session_state` 中是否存在 `"client"` 键。如果不存在，则进行 OpenAI 客户端的初始化。
    *   **第 127 行:** `api_key = os.getenv('OPENAI_API_KEY')` 从环境变量中获取 `OPENAI_API_KEY`。
    *   **第 128-130 行:** `if not api_key:` 如果 `api_key` 不存在（即环境变量未设置），则使用 `st.error()` 显示一个错误消息，并使用 `st.stop()` 停止 Streamlit 应用的执行。
    *   **第 131 行:** `st.session_state.client = OpenAI(api_key=api_key, base_url="https://api.fe8.cn/v1")` 使用获取到的 `api_key` 和指定的 `base_url` 初始化 `OpenAI` 客户端，并将其存储在 `session_state.client` 中，以便在整个应用中复用。

```python
134	st.markdown("""
135	# 🎫 门票助手 (OpenAI 版本)
136	
137	欢迎使用门票助手！我可以帮您查询门票订单信息、统计销售数据等。
138	
139	**功能特性：**
140	- 🔍 智能查询门票订单数据
141	- 📊 生成销售统计报告
142	- 💬 自然语言交互
143	- 🔄 支持多轮对话
144	""")
```
**解析:**
*   **第 134-144 行:** 使用 `st.markdown()` 函数在 Streamlit 页面上显示 Markdown 格式的文本。这部分内容是应用的标题和简介，包括欢迎语和功能特性列表。

```python
147	with st.sidebar:
148	    st.markdown("### ⚙️ 设置")
149	
150	    model = st.selectbox(
151	        "选择模型",
152	        ["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"],
153	        index=0
154	    )
155	
156	    temperature = st.slider(
157	        "温度 (Creativity)",
158	        min_value=0.0,
159	        max_value=2.0,
160	        value=0.7,
161	        step=0.1
162	    )
163	
164	    max_tokens = st.slider(
165	        "最大令牌数",
166	        min_value=256,
167	        max_value=4096,
168	        value=2048,
169	        step=256
170	    )
171	
172	    st.markdown("---")
173	
174	    st.markdown("### 💡 建议问题")
175	    suggestions = [
176	        "2023年4、5、6月一日门票，二日门票的销量多少？帮我按照周进行统计",
177	        "2023年7月的不同省份的入园人数统计",
178	        "帮我查看2023年10月1-7日销售渠道订单金额排名",
179	    ]
180	
181	    for i, suggestion in enumerate(suggestions):
182	        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
183	            st.session_state.messages.append({
184	                "role": "user",
185	                "content": suggestion
186	            })
187	            st.rerun()
188	
189	    st.markdown("---")
190	
191	    if st.button("🗑️ 清空对话历史", use_container_width=True):
192	        st.session_state.messages = []
193	        st.rerun()
```
**解析:**
*   **第 147 行:** `with st.sidebar:` 创建一个上下文管理器，所有在此块内的 Streamlit 组件都将显示在应用的侧边栏中。
*   **第 148 行:** `st.markdown("### ⚙️ 设置")` 在侧边栏中显示一个三级标题“设置”。
*   **第 150-154 行:** `st.selectbox()` 创建一个下拉选择框，用于选择 OpenAI 模型。
    *   `"选择模型"`: 下拉框的标签。
    *   `["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"]`: 可供选择的模型列表。
    *   `index=0`: 默认选中列表中的第一个模型。
*   **第 156-162 行:** `st.slider()` 创建一个滑块，用于调整模型的“温度”（Creativity，即生成文本的随机性）。
    *   `"温度 (Creativity)"`: 滑块的标签。
    *   `min_value=0.0`, `max_value=2.0`, `value=0.7`, `step=0.1`: 设置滑块的最小值、最大值、默认值和步长。
*   **第 164-170 行:** `st.slider()` 创建另一个滑块，用于调整模型生成文本的最大令牌数。
    *   `"最大令牌数"`: 滑块的标签。
    *   `min_value=256`, `max_value=4096`, `value=2048`, `step=256`: 设置滑块的最小值、最大值、默认值和步长。
*   **第 172 行:** `st.markdown("---")` 在侧边栏中显示一条水平分隔线。
*   **第 174 行:** `st.markdown("### 💡 建议问题")` 在侧边栏中显示一个三级标题“建议问题”。
*   **第 175-179 行:** 定义了一个 `suggestions` 列表，包含一些预设的建议问题。
*   **第 181-187 行:** 遍历 `suggestions` 列表，为每个建议问题创建一个 Streamlit 按钮。
    *   `if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):` 当用户点击某个建议问题按钮时。
        *   `st.session_state.messages.append(...)`: 将点击的建议问题作为用户消息添加到聊天历史中。
        *   `st.rerun()`: 重新运行 Streamlit 应用，以更新聊天界面并触发模型响应。
*   **第 189 行:** `st.markdown("---")` 在侧边栏中显示另一条水平分隔线。
*   **第 191-193 行:** 创建一个“清空对话历史”按钮。
    *   `if st.button("🗑️ 清空对话历史", use_container_width=True):` 当用户点击此按钮时。
        *   `st.session_state.messages = []`: 清空聊天历史列表。
        *   `st.rerun()`: 重新运行 Streamlit 应用，以更新聊天界面。

```python
196	st.markdown("### 💬 对话")
197	
198	# 显示对话历史
199	for message in st.session_state.messages:
200	    with st.chat_message(message["role"]):
201	        st.markdown(message["content"])
202	
203	# 用户输入
204	user_input = st.chat_input("请输入您的问题...")
205	
206	if user_input:
207	    # 添加用户消息
208	    st.session_state.messages.append({
209	        "role": "user",
210	        "content": user_input
211	    })
212	
213	    # 显示用户消息
214	    with st.chat_message("user"):
215	        st.markdown(user_input)
```
**解析:**
*   **第 196 行:** `st.markdown("### 💬 对话")` 在主内容区域显示一个三级标题“对话”。
*   **第 199-201 行:** 遍历 `st.session_state.messages` 中存储的聊天历史记录。
    *   `with st.chat_message(message["role"]):` 使用 `st.chat_message` 创建一个聊天气泡，根据消息的 `role`（“user”或“assistant”）显示不同的样式。
    *   `st.markdown(message["content"])`: 在聊天气泡中显示消息内容。
*   **第 204 行:** `user_input = st.chat_input("请输入您的问题...")` 创建一个聊天输入框，提示文本为“请输入您的问题...”，用户输入的内容将赋值给 `user_input` 变量。
*   **第 206 行:** `if user_input:` 如果用户输入了内容（即 `user_input` 不为空）。
*   **第 208-211 行:** 将用户输入的消息作为一个字典（包含 `role` 为“user”和 `content` 为用户输入）添加到 `st.session_state.messages` 列表中，更新聊天历史。
*   **第 214-215 行:** 使用 `st.chat_message("user")` 显示用户刚刚输入的消息。

```python
217	    # 处理 OpenAI 响应
218	    with st.chat_message("assistant"):
219	        with st.spinner("正在处理您的请求..."):
220	            try:
221	                # 构建消息列表
222	                messages = [
223	                    {"role": "system", "content": SYSTEM_PROMPT},
224	                    *st.session_state.messages
225	                ]
226	
227	                # 调用 OpenAI API
228	                response = st.session_state.client.chat.completions.create(
229	                    model=model,
230	                    messages=messages,
231	                    tools=TOOLS,
232	                    tool_choice="auto",
233	                    temperature=temperature,
234	                    max_tokens=max_tokens
235	                )
236	
237	                assistant_message = response.choices[0].message
238	
239	                # 处理工具调用
240	                if assistant_message.tool_calls:
241	                    # 执行工具调用
242	                    tool_results = []
243	                    for tool_call in assistant_message.tool_calls:
244	                        tool_name = tool_call.function.name
245	                        tool_input = json.loads(tool_call.function.arguments)
246	                        tool_result = process_tool_call(tool_name, tool_input)
247	                        tool_results.append(tool_result)
248	
249	                    # 再次调用 API 以获得最终回复
250	                    messages.append({
251	                        "role": "assistant",
252	                        "content": assistant_message.content or "",
253	                        "tool_calls": [
254	                            {
255	                                "id": tc.id,
256	                                "type": "function",
257	                                "function": {
258	                                    "name": tc.function.name,
259	                                    "arguments": tc.function.arguments
260	                                }
261	                            }
262	                            for tc in assistant_message.tool_calls
263	                        ]
264	                    })
265	
266	                    for tool_result in tool_results:
267	                        messages.append({
268	                            "role": "user",
269	                            "content": tool_result
270	                        })
271	
272	                    final_response = st.session_state.client.chat.completions.create(
273	                        model=model,
274	                        messages=messages,
275	                        temperature=temperature,
276	                        max_tokens=max_tokens
277	                    )
278	
279	                    final_message = final_response.choices[0].message.content
280	                    st.markdown(final_message)
281	
282	                    # 添加到消息历史
283	                    st.session_state.messages.append({
284	                        "role": "assistant",
285	                        "content": final_message
286	                    })
287	                else:
288	                    # 直接返回文本回复
289	                    reply = assistant_message.content or "无法生成回复"
290	                    st.markdown(reply)
291	
292	                    # 添加到消息历史
293	                    st.session_state.messages.append({
294	                        "role": "assistant",
295	                        "content": reply
296	                    })
297	
298	            except APIError as e:
299	                st.error(f"❌ API 错误: {str(e)}")
300	            except Exception as e:
301	                st.error(f"❌ 发生错误: {str(e)}")
```
**解析:**
*   **第 218 行:** `with st.chat_message("assistant"):` 创建一个用于显示助手回复的聊天气泡。
*   **第 219 行:** `with st.spinner("正在处理您的请求..."):` 在处理请求期间显示一个加载动画和文本“正在处理您的请求...”。
*   **第 220 行:** `try:` 语句块开始，用于捕获 OpenAI API 调用和工具执行过程中可能发生的异常。
*   **第 222-224 行:** 构建发送给 OpenAI API 的消息列表 `messages`。
    *   `{"role": "system", "content": SYSTEM_PROMPT}`: 将之前定义的 `SYSTEM_PROMPT` 作为系统消息添加到消息列表的开头，为模型提供上下文和指导。
    *   `*st.session_state.messages`: 使用星号解包操作符，将 `session_state` 中存储的所有历史消息（用户和助手）添加到消息列表中。
*   **第 228-235 行:** 调用 `st.session_state.client.chat.completions.create()` 方法，向 OpenAI API 发送聊天完成请求。
    *   `model=model`: 使用侧边栏选择的模型。
    *   `messages=messages`: 使用构建好的消息列表。
    *   `tools=TOOLS`: 传入之前定义的 `TOOLS` 列表，使模型能够调用这些工具。
    *   `tool_choice="auto"`: 允许模型自动决定是否调用工具。
    *   `temperature=temperature`: 使用侧边栏设置的温度参数。
    *   `max_tokens=max_tokens`: 使用侧边栏设置的最大令牌数。
*   **第 237 行:** `assistant_message = response.choices[0].message` 从 API 响应中提取助手的消息对象。
*   **第 240 行:** `if assistant_message.tool_calls:` 检查助手消息中是否包含工具调用（即模型决定调用了某个工具）。
    *   **第 242 行:** `tool_results = []` 初始化一个空列表，用于存储工具调用的结果。
    *   **第 243-247 行:** 遍历助手消息中的所有工具调用。
        *   `tool_name = tool_call.function.name`: 获取工具的名称。
        *   `tool_input = json.loads(tool_call.function.arguments)`: 解析工具的参数（通常是 JSON 字符串）为 Python 字典。
        *   `tool_result = process_tool_call(tool_name, tool_input)`: 调用 `process_tool_call` 函数来执行实际的工具操作，并将结果存储在 `tool_result` 中。
        *   `tool_results.append(tool_result)`: 将工具执行结果添加到 `tool_results` 列表中。
    *   **第 250-264 行:** 将助手的工具调用消息添加到 `messages` 列表中，以便在后续的 API 调用中提供给模型作为上下文。这包括工具调用的 `role`、`content` (可能为空) 和 `tool_calls` 详细信息。
    *   **第 266-270 行:** 遍历 `tool_results` 列表，将每个工具的执行结果作为用户消息（`role`: "user"）添加到 `messages` 列表中。这模拟了用户向模型提供工具执行结果的场景。
    *   **第 272-276 行:** 再次调用 `st.session_state.client.chat.completions.create()` 方法，这次的 `messages` 列表包含了原始对话、助手的工具调用信息以及工具的执行结果。模型会根据这些信息生成最终的自然语言回复。
    *   **第 279 行:** `final_message = final_response.choices[0].message.content` 从第二次 API 响应中提取最终的助手回复内容。
    *   **第 280 行:** `st.markdown(final_message)` 在 Streamlit 界面上显示最终的助手回复。
    *   **第 283-286 行:** 将最终的助手回复添加到 `st.session_state.messages` 列表中，更新聊天历史。
*   **第 287 行:** `else:` 如果助手消息中不包含工具调用（即模型直接生成了文本回复）。
    *   **第 289 行:** `reply = assistant_message.content or "无法生成回复"` 获取助手的文本回复内容，如果内容为空，则默认为“无法生成回复”。
    *   **第 290 行:** `st.markdown(reply)` 在 Streamlit 界面上显示助手的文本回复。
    *   **第 293-296 行:** 将助手的文本回复添加到 `st.session_state.messages` 列表中，更新聊天历史。
*   **第 298 行:** `except APIError as e:` 捕获 `openai.APIError` 异常。
*   **第 299 行:** `st.error(f"❌ API 错误: {str(e)}")` 在 Streamlit 界面上显示 API 错误信息。
*   **第 300 行:** `except Exception as e:` 捕获所有其他类型的异常。
*   **第 301 行:** `st.error(f"❌ 发生错误: {str(e)}")` 在 Streamlit 界面上显示通用错误信息。

```python
304	st.markdown("---")
305	st.markdown("""
306	<div style="text-align: center; color: #666; font-size: 12px;">
307	    门票助手 v1.0 | 由 OpenAI API 驱动 | 
308	    <a href="https://openai.com" target="_blank">OpenAI</a>
309	</div>
310	""", unsafe_allow_html=True)
```
**解析:**
*   **第 304 行:** `st.markdown("---")` 在页面底部显示一条水平分隔线。
*   **第 305-310 行:** 使用 `st.markdown()` 显示一个页脚信息。`unsafe_allow_html=True` 参数允许 Streamlit 渲染包含 HTML 标签的 Markdown 字符串，这里用于设置文本居中、颜色、字体大小，并包含一个指向 OpenAI 官网的链接。
