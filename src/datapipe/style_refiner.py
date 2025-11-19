import json
from typing import Dict, Any
from openai import OpenAI
import pandas as pd


def refine_chart_style(
    meta: Dict[str, Any],
    chosen_question: str,
    chart_spec: Dict[str, Any],
    agg_df: pd.DataFrame,
    client: OpenAI,
) -> Dict[str, Any]:
    """
    Take the existing chart_spec + metadata + question + aggregated data,
    and ask the LLM to propose styling improvements.

    It will return a NEW chart spec:
    - keep existing structural fields (chart_type, x, y, aggregation, group_by, filters)
    - add a 'style' field for visual polish (title, labels, color suggestions, etc.)
    """

    # To avoid sending huge data, only pass a small preview
    data_preview = agg_df.head(20).to_dict(orient="records")

    prompt = f"""
You are a professional data visualization designer.

Here is the dataset metadata:
{json.dumps(meta, indent=2)}

The user selected this question:
{chosen_question}

Here is the current chart specification:
{json.dumps(chart_spec, indent=2)}

Here is the aggregated data used for this chart (preview):
{json.dumps(data_preview, indent=2)}

Please refine the chart design to make it clearer and more visually appealing.

Rules:
- Keep the existing structural fields: chart_type, x, y, aggregation, group_by, filters.
- Add a new field called "style" with properties such as:
  - title: a clear, human-readable chart title
  - subtitle: (optional) a short one-line explanation
  - x_label: label for the x-axis
  - y_label: label for the y-axis
  - color_palette: a short description or name of a palette (e.g., "blues", "category10")
  - legend_position: e.g., "top-right", "bottom", or "none"
  - annotations: a list of short text annotations for notable points (optional)

Return ONLY a single JSON object representing the refined chart spec.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    refined_spec = json.loads(response.choices[0].message.content)
    return refined_spec