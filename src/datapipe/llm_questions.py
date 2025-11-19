import json
from typing import Dict, Any
from openai import OpenAI


def suggest_questions_from_meta(meta: Dict[str, Any], client: OpenAI) -> str:
    """
    用 dataset meta 让 LLM 生成 5 个问题（纯文本）
    """
    prompt = f"""
You are a data analysis assistant.
Based on the following metadata, generate exactly 5 questions
that a beginner user might want to ask about this dataset.

Metadata:
{json.dumps(meta, indent=2)}

Rules:
- Questions must be simple and understandable for beginners.
- Questions must be directly solvable using the dataset.
- No explanation. Only list the 5 questions.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    # ✅ 注意新版 SDK：用 .message.content
    return response.choices[0].message.content