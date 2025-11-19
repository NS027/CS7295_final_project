# src/datapipe/insight.py
import json
import pandas as pd
from openai import OpenAI
from typing import Dict, Any, List

def generate_insights(
    meta: Dict[str, Any],
    chart_spec: Dict[str, Any],
    agg_df: pd.DataFrame,
    client: OpenAI,
    max_insights: int = 3,
) -> str:
    """
    根据元数据 + chart_spec + 聚合后的数据，生成自然语言分析。
    返回一段 markdown 文本。
    """
    # 把聚合后的数据压缩成一个小表丢给 LLM（避免超长）
    table_preview = agg_df.to_dict(orient="records")

    prompt = f"""
You are a senior data analyst.

Here is the dataset metadata:

{json.dumps(meta, indent=2)}

Here is the chart specification:

{json.dumps(chart_spec, indent=2)}

Here is the aggregated data used for the chart (as JSON rows):

{json.dumps(table_preview, indent=2)}

Please provide up to {max_insights} clear, concise insights
about this chart and the underlying data. Focus on patterns, comparisons,
and notable highs/lows. Write the answer in bullet points.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content