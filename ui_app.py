import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.datapipe.json_schema import csv_bytes_to_llm_json
from src.datapipe.llm_questions import suggest_questions_from_meta
from src.datapipe.llm_chart_spec import generate_chart_spec_from_question
from src.datapipe.viz_executor import aggregate_dataframe
from src.datapipe.insight import generate_insights
from src.datapipe.style_refiner import refine_chart_style

# Load .env for API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="LLM Data Visualization Assistant", layout="wide")
st.title("📊🤖 LLM-Powered Data Visualization Assistant")

st.write("Upload a CSV file and I will help you:")
st.markdown(
"""
1. Analyze its structure and generate JSON metadata  
2. Use an LLM to suggest 5 beginner-friendly analytical questions  
3. Let you choose one question  
4. Generate a chart specification (chart spec)  
5. Aggregate data and produce a visualization  
6. Generate insights based on the chart  
"""
)

# ====================================================
# 1. Upload CSV
# ====================================================
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Read DataFrame
file_bytes = uploaded_file.getvalue()
df = pd.read_csv(uploaded_file)

st.subheader("👀 Data Preview")
st.dataframe(df.head())

# Generate metadata
dataset_json = csv_bytes_to_llm_json(file_bytes, dataset_name=uploaded_file.name)
meta = dataset_json["meta"]

with st.expander("Show JSON metadata"):
    st.json(meta)

# ====================================================
# 2. Generate LLM questions
# ====================================================
st.subheader("❓ Auto-generated Analysis Questions")

if "questions" not in st.session_state:
    st.session_state["questions"] = None

if st.button("✨ Generate 5 questions"):
    with st.spinner("LLM is generating questions..."):
        questions_text = suggest_questions_from_meta(meta, client)
        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        st.session_state["questions"] = questions
        st.session_state["questions_text"] = questions_text

if st.session_state["questions"] is None:
    st.info("Click the button above to generate questions.")
    st.stop()

questions = st.session_state["questions"]

for i, q in enumerate(questions, start=1):
    st.markdown(f"**{i}. {q}**")

chosen_question = st.radio(
    "👉 Select a question:",
    options=questions,
    index=0,
)

# ====================================================
# 3. Generate Chart + Insight
# ====================================================
st.subheader("📊 Visualization + 🔍 Insights")

# Ensure some keys exist in session_state
for key in ["chart_spec", "df_agg", "meta", "chosen_question", "insights_md"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.button("🚀 Generate Chart and Insights"):
    # 3.1 LLM generates chart spec
    with st.spinner("LLM is generating chart specification..."):
        chart_spec = generate_chart_spec_from_question(meta, chosen_question, client)

    # store to session_state
    st.session_state["chart_spec"] = chart_spec
    st.session_state["chosen_question"] = chosen_question
    st.session_state["meta"] = meta

    # 3.2 Aggregate Data
    with st.spinner("Aggregating data and drawing chart..."):
        df_agg = aggregate_dataframe(df, chart_spec)
        st.session_state["df_agg"] = df_agg

# ---- Show chart + insights if we have them in session_state ----
if st.session_state["chart_spec"] is not None and st.session_state["df_agg"] is not None:
    chart_spec = st.session_state["chart_spec"]
    df_agg = st.session_state["df_agg"]

    st.markdown("### Generated Chart Spec")
    st.json(chart_spec)

    chart_type = chart_spec.get("chart_type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    agg = chart_spec.get("aggregation", "none")
    value_col = f"{agg}_{y}" if agg != "none" else y

    st.markdown("### Aggregated Data (Used for plotting)")
    st.dataframe(df_agg)

    # Visualization
    st.markdown("### 📈 Visualization Preview")

    if value_col not in df_agg.columns:
        st.error(
            f"The column '{value_col}' is missing from aggregated data. "
            f"Current columns: {df_agg.columns.tolist()}"
        )
    else:
        plot_df = df_agg.copy()

        if chart_type == "bar":
            st.bar_chart(plot_df.set_index(x)[value_col])
        elif chart_type == "line":
            st.line_chart(plot_df.set_index(x)[value_col])
        elif chart_type == "scatter":
            try:
                scatter_df = plot_df[[x, value_col]].copy()
                scatter_df[x] = pd.to_numeric(scatter_df[x], errors="coerce")
                st.scatter_chart(scatter_df, x=x, y=value_col)
            except Exception as e:
                st.error(f"Scatter plot failed: {e}")
        elif chart_type == "histogram":
            hist_values = plot_df[value_col].dropna()
            counts, bins = pd.cut(hist_values, bins=10, retbins=True)
            hist_df = counts.value_counts().sort_index()
            hist_plot_df = pd.DataFrame({
                "bin": hist_df.index.astype(str),
                "count": hist_df.values,
            }).set_index("bin")
            st.bar_chart(hist_plot_df)
        else:
            st.error(f"Unsupported chart type: {chart_type}")

    # 3.3 Generate Insights (only generate once, then reuse)
    if st.session_state["insights_md"] is None:
        with st.spinner("LLM generating insights..."):
            insights_md = generate_insights(
                st.session_state["meta"],
                chart_spec,
                df_agg,
                client,
            )
            st.session_state["insights_md"] = insights_md

    st.markdown("### 🔍 Insights")
    st.markdown(st.session_state["insights_md"])

# ====================================================
# 4. Refine Chart Style (separate section)
# ====================================================
st.subheader("🎨 Optional: Refine chart style with LLM")

if st.session_state["chart_spec"] is None or st.session_state["df_agg"] is None:
    st.info("Generate a chart first (above), then you can refine its style here.")
else:
    if st.button("💅 Refine chart style"):
        with st.spinner("LLM is refining chart style..."):
            refined_spec = refine_chart_style(
                st.session_state["meta"],
                st.session_state["chosen_question"],
                st.session_state["chart_spec"],
                st.session_state["df_agg"],
                client,
            )
            st.session_state["refined_chart_spec"] = refined_spec

    if "refined_chart_spec" in st.session_state and st.session_state["refined_chart_spec"] is not None:
        st.markdown("### 🎨 Refined Chart Spec (with style)")
        st.json(st.session_state["refined_chart_spec"])