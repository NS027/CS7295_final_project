import os
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

from src.datapipe.json_schema import csv_bytes_to_llm_json
from src.datapipe.viz_executor import execute_chart_spec
from src.datapipe.insight import generate_insights
from src.datapipe.llm_questions import suggest_questions_from_meta
from src.datapipe.llm_chart_spec import generate_chart_spec_from_question

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# 方便以后接前端（React / Vue 等）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/questions")
async def get_questions(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_bytes = await file.read()
    dataset_json = csv_bytes_to_llm_json(file_bytes, dataset_name=file.filename)
    meta = dataset_json["meta"]

    questions_text = suggest_questions_from_meta(meta, client)
    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

    return {
        "dataset_meta": meta,
        "questions_text": questions_text,
        "questions": questions,
    }


@app.post("/chart")
async def get_chart_and_insights(
    file: UploadFile = File(...),
    chosen_question: str = Form(...),
) -> Dict[str, Any]:
    """
    Step 2:
    用户重新上传同一个 CSV + 选择的问题 -> 返回 chart_spec + 聚合后的数据 + insight 文本
    （前端可用 chart_spec 自己画图，也可以以后接你现有的 Python 画图）
    """
    file_bytes = await file.read()
    # 读原始数据
    tmp_path = Path(file.filename)
    tmp_path.write_bytes(file_bytes)
    df = pd.read_csv(tmp_path)

    dataset_json = csv_bytes_to_llm_json(file_bytes, dataset_name=file.filename)
    meta = dataset_json["meta"]

    # LLM 生成 chart spec
    chart_spec = generate_chart_spec_from_question(meta, chosen_question, client)

    # 根据 spec 聚合 + 画图（当前 API 只返回数据，不返回图片）
    agg_df = execute_chart_spec(df, chart_spec)

    # 生成简单 insight（几条 bullet point 就够）
    insights_md = generate_insights(meta, chart_spec, agg_df, client)

    # agg_df 不能直接 JSON，转 dict
    agg_data = agg_df.to_dict(orient="records")

    return {
        "dataset_meta": meta,
        "chosen_question": chosen_question,
        "chart_spec": chart_spec,
        "agg_data": agg_data,
        "insights_md": insights_md,
    }