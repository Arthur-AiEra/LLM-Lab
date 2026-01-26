#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals RAG_HELPFULNESS_PROMPT - RAG 帮助性评估示例
"""

from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_HELPFULNESS_PROMPT

from app_config import eval_llm

evaluator = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("RAG_HELPFULNESS_PROMPT - RAG 帮助性评估测试")
print("=" * 60)

test_cases = [
    {
        "name": "有帮助的回答",
        "inputs": "什么是量子计算？",
        "outputs": "量子计算是一种利用量子力学原理进行计算的技术。它使用量子比特（qubit）来存储和处理信息，相比传统计算机具有指数级的计算优势。"
    },
    {
        "name": "部分有帮助",
        "inputs": "什么是量子计算？",
        "outputs": "量子计算是一种计算技术。"
    },
    {
        "name": "无帮助的回答",
        "inputs": "什么是量子计算？",
        "outputs": "计算机有很多种类型。"
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

