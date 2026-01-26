#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals ANSWER_RELEVANCE_PROMPT - 相关性评估示例
"""

from openevals.llm import create_llm_as_judge
from openevals.prompts import ANSWER_RELEVANCE_PROMPT

from app_config import eval_llm

evaluator = create_llm_as_judge(
    prompt=ANSWER_RELEVANCE_PROMPT,
    feedback_key="relevance",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("ANSWER_RELEVANCE_PROMPT - 相关性评估测试")
print("=" * 60)

test_cases = [
    {
        "name": "高度相关",
        "inputs": "什么是机器学习？",
        "outputs": "机器学习是让计算机从数据中学习的技术。"
    },
    {
        "name": "部分相关",
        "inputs": "什么是机器学习？",
        "outputs": "人工智能是一个广泛的领域。"
    },
    {
        "name": "不相关",
        "inputs": "什么是机器学习？",
        "outputs": "今天天气很好。"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n【测试 {i}: {test_case['name']}】")
    print(f"输入: {test_case['inputs']}")
    print(f"输出: {test_case['outputs']}")
    
    try:
        result = evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
        )
        
        score = result.get("score") if isinstance(result, dict) else result
        print(f"[OK] 评估分数: {score:.3f}")
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

