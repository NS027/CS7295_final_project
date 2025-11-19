import json
from typing import Dict, Any
from openai import OpenAI


def generate_chart_spec_from_question(
    meta: Dict[str, Any],
    question: str,
    client: OpenAI,
) -> Dict[str, Any]:
    """
    用 dataset meta + 用户提问，让 LLM 生成 chart spec。
    chart spec 必须能被 execute_chart_spec() 执行。
    """

    prompt = f"""
You are a data visualization assistant.
Given the dataset metadata and the user's question,
generate a chart specification in JSON format.

Metadata:
{json.dumps(meta, indent=2)}

User Question:
"{question}"

Chart Spec Rules:
- Output JSON only.
- Must include keys: chart_type, x, y, aggregation, group_by, filters.
- filters must be a list of objects like:
  {{"column":"price","op":">","value":100}}.
- chart_type must be one of: "bar", "line", "scatter", "histogram".
- x must be a column from the dataset.
- y must be a numeric column from the dataset.
- aggregation must be: "sum", "mean", "count", or "none".
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    # ✅ 同样使用 .message.content
    return json.loads(response.choices[0].message.content)