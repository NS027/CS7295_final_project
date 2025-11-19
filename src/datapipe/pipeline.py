# src/datapipe/pipeline.py
from typing import Dict, Any, Tuple, List
from openai import OpenAI
import pandas as pd
from pathlib import Path

from .json_schema import csv_bytes_to_llm_json
from .viz_executor import execute_chart_spec
from .insight import generate_insights


def run_full_analysis(
    csv_path: Path,
    client: OpenAI,
    question_index: int = 1,
) -> Dict[str, Any]:
    """
    整个 pipeline：
    1. 读 CSV → JSON meta
    2. LLM 生成 5 个问题
    3. 用 question_index 选一个问题
    4. LLM 生成 chart_spec
    5. 执行 chart_spec 画图，得到 agg_df
    6. 基于结果生成 insight

    返回一个 dict，里面存所有中间结果，方便 API / UI 使用。
    """
    # 读 CSV
    df = pd.read_csv(csv_path)
    file_bytes = csv_path.read_bytes()
    dataset_json = csv_bytes_to_llm_json(file_bytes, dataset_name=csv_path.name)
    meta = dataset_json["meta"]

    # 1) 生成问题
    from .llm_questions import suggest_questions_from_meta  # 你可以把 notebook 里的函数移到这
    questions_text = suggest_questions_from_meta(meta, client)

    # 简单解析成每行一个问题
    questions = [line.strip() for line in questions_text.split("\n") if line.strip()]
    chosen_question = questions[question_index - 1]

    # 2) 生成 chart_spec
    from .llm_chart_spec import generate_chart_spec_from_question  # 同理移到模块
    chart_spec = generate_chart_spec_from_question(meta, chosen_question, client)

    # 3) 画图 & 聚合
    agg_df = execute_chart_spec(df, chart_spec)

    # 4) insight
    insights_md = generate_insights(meta, chart_spec, agg_df, client)

    return {
        "dataset_meta": meta,
        "questions_text": questions_text,
        "questions": questions,
        "chosen_question": chosen_question,
        "chart_spec": chart_spec,
        "agg_df": agg_df,
        "insights_md": insights_md,
    }