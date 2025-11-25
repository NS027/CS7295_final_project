import json
from typing import Dict, Any
from openai import OpenAI


def suggest_questions_from_meta(meta: Dict[str, Any], client: OpenAI) -> str:
    """
    用 dataset meta 让 LLM 生成 N 个问题（纯文本）
    """
    QUESTION_NUM = 10 # modify this to configure the number of questions we want to generate

    user_prompt = f"""The number of GOALS to generate is {QUESTION_NUM}. The goals should be based on the data summary below, \n\n .
{json.dumps(meta, indent=2)} \n\n
\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a data analyst persona, who is insterested in complex, insightful goals about the data. \n
However, the description of questions should be simple and understandable for beginners\n

Rules:
- Questions must be sophisticated and insightful for the goal of data analysis
- However, the description of questions should remain simple and understandable for beginners.
- Questions must be directly solvable using the dataset.
- No explanation. Only list the questions."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_prompt}],
    )

    # ✅ 注意新版 SDK：用 .message.content
    return response.choices[0].message.content