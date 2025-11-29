import json
from typing import Dict, Any
from openai import OpenAI


def suggest_questions_from_meta(meta: Dict[str, Any], client: OpenAI) -> list[str]:
    """
    用 dataset meta 让 LLM 生成 N 个问题（返回 JSON list）
    """
    QUESTION_NUM = 20 

    user_prompt = f"""The number of GOALS to generate is {QUESTION_NUM}. The goals should be based on the data summary below, \n\n .
{json.dumps(meta, indent=2)} \n\n
\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a data analyst persona, who is insterested in complex, insightful goals about the data. \n
However, the description of questions should be simple and understandable for beginners\n

Rules:
- Questions must be sophisticated and insightful for the goal of data analysis
- However, the description of questions should remain simple and understandable for beginners.
- Questions must be directly solvable using the dataset.
- Output pure JSON only: a list of strings. ["Question 1", "Question 2", ...]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        # 尝试解析 JSON
        data = json.loads(content)
        # 兼容 { "questions": [...] } 或直接 [...]
        if isinstance(data, dict):
             # 找任何看起来像 list 的 value
            for k, v in data.items():
                if isinstance(v, list):
                    return v
            return list(data.values())[0] if data else []
        elif isinstance(data, list):
            return data
        else:
            return []
    except json.JSONDecodeError:
        # Fallback: 按行分割
        return [q.strip() for q in content.split("\n") if q.strip()]