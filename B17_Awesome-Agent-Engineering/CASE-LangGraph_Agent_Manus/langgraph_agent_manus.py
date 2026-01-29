"""
基于 LangGraph 的 OpenManus 风格 Agent 示例

核心特性：
1. 思考-执行-观察 循环
2. StateGraph 定义节点
3. Human-in-the-loop（高风险操作需人工确认）
4. 真实文件系统操作

依赖：pip install langgraph langchain-openai
"""

import os
import operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


# 工作目录（当前脚本所在目录）
WORK_DIR = Path(__file__).parent


# ============ 1. 定义状态 ============
class AgentState(TypedDict):
    task: str                                           # 用户任务
    plan: str                                           # 规划结果
    action: str                                         # 待执行的动作
    is_high_risk: bool                                  # 是否高风险
    human_approved: bool                                # 人工确认
    observation: str                                    # 执行结果
    messages: Annotated[list, operator.add]             # 消息历史
    step_count: int                                     # 步数计数
    is_complete: bool                                   # 是否完成


# ============ 2. 定义节点 ============

# 使用 DashScope 的 qwen-flash 模型
llm = ChatOpenAI(
    model="qwen-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def planner_node(state: AgentState) -> AgentState:
    """规划节点：分析任务，决定下一步动作"""
    task = state["task"]
    observation = state.get("observation", "")
    step_count = state.get("step_count", 0)

    prompt = f"""
你是一个文件管理助手，工作目录是: {WORK_DIR}

用户任务: {task}
已执行步数: {step_count}
上一步执行结果: {observation if observation else "无"}

可用动作：
- list_files          （列出当前目录文件）
- read_file: 文件名   （读取文件内容）
- delete_file: 文件名 （删除文件，高风险）
- create_file: 文件名 （创建文件）
- DONE                （任务已完成，结束）

重要规则：
1. 如果用户只是想查看/列出文件，执行 list_files 后就回复 DONE
2. 如果上一步的执行结果已经满足了用户需求，直接回复 DONE
3. 不要重复执行相同的操作
4. 只回复动作名称，不要加 - 符号，不要解释

现在请回复下一步动作："""

    response = llm.invoke(prompt)
    action = response.content.strip()

    # 判断是否高风险操作
    high_risk_keywords = ["delete", "remove", "drop", "truncate"]
    is_high_risk = any(keyword in action.lower() for keyword in high_risk_keywords)

    print(f"\n[Planner] 决策: {action}")
    if is_high_risk:
        print(f"[Planner] 检测到高风险操作!")

    return {
        **state,
        "plan": f"计划执行: {action}",
        "action": action,
        "is_high_risk": is_high_risk,
        "step_count": step_count + 1,
    }


def human_review_node(state: AgentState) -> AgentState:
    """
    人工审核节点 - 真实等待用户输入
    """
    print("\n" + "=" * 50)
    print("[Human Review Required]")
    print("=" * 50)
    print(f"检测到高风险操作: {state['action']}")
    print(f"工作目录: {WORK_DIR}")
    print("-" * 50)

    while True:
        user_input = input("是否批准执行？(y/n): ").strip().lower()
        if user_input in ["y", "yes"]:
            print("[Human] 已批准执行")
            return {**state, "human_approved": True}
        elif user_input in ["n", "no"]:
            print("[Human] 已拒绝执行")
            return {
                **state,
                "human_approved": False,
                "action": "DONE",
                "observation": "用户拒绝了高风险操作，任务终止"
            }
        else:
            print("请输入 y 或 n")


def executor_node(state: AgentState) -> AgentState:
    """执行节点：执行真实的文件系统操作"""
    action = state["action"]

    print(f"\n[Executor] 正在执行: {action}")

    try:
        if "list_files" in action.lower():
            # 列出目录文件
            files = list(WORK_DIR.iterdir())
            file_list = []
            for f in files:
                if f.is_file():
                    size = f.stat().st_size
                    file_list.append(f"  - {f.name} ({size} bytes)")
                else:
                    file_list.append(f"  - {f.name}/ (目录)")
            observation = f"文件列表 ({WORK_DIR}):\n" + "\n".join(file_list) if file_list else "目录为空"

        elif "read_file" in action.lower():
            # 读取文件
            file_name = action.split(":")[-1].strip() if ":" in action else ""
            file_path = WORK_DIR / file_name
            if file_path.exists() and file_path.is_file():
                content = file_path.read_text(encoding="utf-8")[:500]  # 最多读500字符
                observation = f"文件内容 ({file_name}):\n{content}"
            else:
                observation = f"文件不存在: {file_name}"

        elif "delete_file" in action.lower():
            # 删除文件
            file_name = action.split(":")[-1].strip() if ":" in action else ""
            file_path = WORK_DIR / file_name
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                observation = f"已成功删除文件: {file_name}"
            else:
                observation = f"文件不存在或无法删除: {file_name}"

        elif "create_file" in action.lower():
            # 创建文件
            file_name = action.split(":")[-1].strip() if ":" in action else "new_file.txt"
            file_path = WORK_DIR / file_name
            file_path.write_text(f"# 由 LangGraph Agent 创建\n创建时间: {__import__('datetime').datetime.now()}\n", encoding="utf-8")
            observation = f"已成功创建文件: {file_name}"

        elif "DONE" in action.upper():
            observation = state.get("observation", "任务完成")

        else:
            observation = f"未知操作: {action}"

    except Exception as e:
        observation = f"执行出错: {str(e)}"

    print(f"[Executor] 结果: {observation}")

    return {
        **state,
        "observation": observation,
        "messages": [{"role": "assistant", "content": observation}],
    }


def reviewer_node(state: AgentState) -> AgentState:
    """审查节点：判断任务是否完成"""
    action = state["action"]
    observation = state.get("observation", "")
    step_count = state.get("step_count", 0)
    task = state.get("task", "").lower()

    # 任务完成条件
    is_complete = False

    # 1. 明确说 DONE
    if "DONE" in action.upper():
        is_complete = True
    # 2. 超过最大步数
    elif step_count >= 5:
        is_complete = True
    # 3. 简单查看任务：列出文件后就完成
    elif any(word in task for word in ["查看", "列出", "有哪些", "显示"]):
        if "文件列表" in observation:
            is_complete = True

    if is_complete:
        print(f"\n[Reviewer] 任务完成，共执行 {step_count} 步")
    else:
        print(f"\n[Reviewer] 继续执行，当前第 {step_count} 步")

    return {
        **state,
        "is_complete": is_complete,
    }


# ============ 3. 定义路由 ============

def route_after_plan(state: AgentState) -> Literal["human_review", "executor"]:
    """规划后路由：高风险操作需人工确认"""
    if state.get("is_high_risk", False):
        return "human_review"
    return "executor"


def route_after_review(state: AgentState) -> Literal["planner", "__end__"]:
    """审查后路由：未完成则继续循环"""
    if state.get("is_complete", False):
        return END
    return "planner"


# ============ 4. 构建图 ============

def build_agent_graph():
    """构建 Agent 状态图"""

    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("planner", planner_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)

    # 设置入口
    graph.set_entry_point("planner")

    # 添加条件边：规划后判断是否需要人工审核
    graph.add_conditional_edges(
        "planner",
        route_after_plan,
        {
            "human_review": "human_review",
            "executor": "executor",
        }
    )

    # 人工审核后执行
    graph.add_edge("human_review", "executor")

    # 执行后审查
    graph.add_edge("executor", "reviewer")

    # 审查后判断是否继续循环
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "planner": "planner",
            END: END,
        }
    )

    return graph

# ============ 5. 运行 ============

def main():
    """
    运行文件管理 Agent

    使用真实文件系统操作，高风险操作需人工确认
    """
    print("=" * 60)
    print("LangGraph Agent 示例 - Human-in-the-loop")
    print("=" * 60)
    print(f"工作目录: {WORK_DIR}")

    # 构建并编译图
    graph = build_agent_graph()
    app = graph.compile()

    while True:
        print("\n" + "-" * 60)
        print("示例任务:")
        print("  1. 查看当前文件夹下有哪些文件")
        print("  2. 删除1.txt (会触发人工确认)")
        print("  3. 自定义任务")
        print("  4. 退出")

        choice = input("\n选择任务 (1/2/3/4): ").strip()

        if choice == "4":
            print("\n再见!")
            break
        elif choice == "1":
            task = "查看当前文件夹下有哪些文件"
        elif choice == "2":
            task = "删除1.txt文件"
        elif choice == "3":
            task = input("请输入自定义任务: ").strip()
            if not task:
                print("任务不能为空，请重新选择")
                continue
        else:
            print("无效选择，请重新输入")
            continue

        print(f"\n任务: {task}")
        print("-" * 60)

        # 初始状态
        initial_state = {
            "task": task,
            "messages": [],
            "step_count": 0,
            "is_complete": False,
            "human_approved": False,
            "observation": "",
            "action": "",
            "plan": "",
            "is_high_risk": False,
        }

        # 运行 Agent（设置递归限制防止无限循环）
        try:
            result = app.invoke(initial_state, {"recursion_limit": 50})

            print("\n" + "=" * 60)
            print("任务执行完成!")
            print("=" * 60)
            print(f"最终结果: {result.get('observation', 'N/A')}")
            print(f"总步数: {result.get('step_count', 0)}")

        except KeyboardInterrupt:
            print("\n\n用户中断执行")
            break
        except Exception as e:
            print(f"\n执行出错: {e}")


if __name__ == "__main__":
    main()
