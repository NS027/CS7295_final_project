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

import concurrent.futures
import streamlit.components.v1 as components

# Load .env for API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_chart_specs_batch(questions: list[str], meta: dict, client: OpenAI) -> dict:
    """
    Parallel generate chart specs for a batch of questions.
    Returns: { question_text: spec_dict }
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_q = {
            executor.submit(generate_chart_spec_from_question, meta, q, client): q 
            for q in questions
        }
        for future in concurrent.futures.as_completed(future_to_q):
            q = future_to_q[future]
            try:
                spec = future.result()
                results[q] = spec
            except Exception as e:
                print(f"Error generating spec for {q}: {e}")
                results[q] = None
    return results

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
# 2. Generate Dashboard
# ====================================================
st.subheader("📊 Infinite Scroll Dashboard")

# Initialize Session State
if "all_questions" not in st.session_state:
    st.session_state["all_questions"] = []
if "loaded_count" not in st.session_state:
    st.session_state["loaded_count"] = 4
if "chart_specs_map" not in st.session_state:
    st.session_state["chart_specs_map"] = {}

# Start Analysis Button
if not st.session_state["all_questions"]:
    if st.button("🚀 Start Analysis (Generate 20 Questions)"):
        with st.spinner("Brainstorming analysis questions..."):
            questions = suggest_questions_from_meta(meta, client)
            st.session_state["all_questions"] = questions
            # Reset
            st.session_state["loaded_count"] = 4 
            st.session_state["chart_specs_map"] = {}
        st.rerun()

# Main Dashboard Loop
if st.session_state["all_questions"]:
    all_qs = st.session_state["all_questions"]
    current_count = st.session_state["loaded_count"]
    
    # Slice needed questions
    questions_to_show = all_qs[:current_count]
    
    # --------------------------------------------------------
    # 1. RENDER EXISTING CHARTS FIRST (To avoid flicker)
    # --------------------------------------------------------
    cols = st.columns(2)
    
    # We iterate only through what we have specs for, or show placeholder
    for i, question in enumerate(questions_to_show):
        col = cols[i % 2]
        spec = st.session_state["chart_specs_map"].get(question)
        
        with col:
            st.markdown(f"**Q{i+1}. {question}**")
            
            if spec:
                try:
                    df_agg = aggregate_dataframe(df, spec)
                    
                    chart_type = spec.get("chart_type")
                    x = spec.get("x")
                    y = spec.get("y")
                    agg = spec.get("aggregation", "none")
                    value_col = f"{agg}_{y}" if agg != "none" else y
                    
                    # Chart Preview
                    if chart_type == "bar":
                        st.bar_chart(df_agg.set_index(x)[value_col], height=300)
                    elif chart_type == "line":
                        st.line_chart(df_agg.set_index(x)[value_col], height=300)
                    elif chart_type == "scatter":
                        st.scatter_chart(df_agg, x=x, y=value_col, height=300)
                    elif chart_type == "histogram":
                        st.bar_chart(df_agg[value_col], height=300)
                    
                    # Details Expander
                    with st.expander(f"🔍 View Details & Insights"):
                        st.json(spec)
                        st.dataframe(df_agg)
                        
                        # Lazy Insight
                        insight_key = f"insight_{i}"
                        if insight_key not in st.session_state:
                            st.session_state[insight_key] = None
                            
                        if st.button(f"Generate Insight", key=f"btn_insight_{i}"):
                            with st.spinner("Analyzing..."):
                                insight = generate_insights(meta, spec, df_agg, client)
                                st.session_state[insight_key] = insight
                        
                        if st.session_state[insight_key]:
                            st.markdown(st.session_state[insight_key])

                except Exception as e:
                    st.error(f"Chart error: {e}")
            else:
                # Placeholder while loading
                st.info("⏳ Waiting for chart data...")

    # --------------------------------------------------------
    # 2. CHECK & LOAD MISSING SPECS (At bottom, non-blocking visual)
    # --------------------------------------------------------
    missing_spec_qs = [q for q in questions_to_show if q not in st.session_state["chart_specs_map"]]
    
    if missing_spec_qs:
        # Show a spinner at the bottom, NOT clearing previous content
        with st.spinner(f"Loading {len(missing_spec_qs)} more charts..."):
            new_specs = generate_chart_specs_batch(missing_spec_qs, meta, client)
            st.session_state["chart_specs_map"].update(new_specs)
            st.rerun()

    # --------------------------------------------------------
    # 3. INFINITE SCROLL TRIGGER
    # --------------------------------------------------------
    if current_count < len(all_qs) and not missing_spec_qs:
        st.markdown("---")
        
        # We give the button a unique key based on count to force recreation if needed
        btn_key = f"btn_load_more_{current_count}"
        
        if st.button("⬇️ Load Next 4 Charts", key=btn_key, use_container_width=True):
             st.session_state["loaded_count"] += 4
             st.rerun()

        # Improved JS Auto-Clicker
        components.html(
            f"""
            <script>
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        const buttons = window.parent.document.querySelectorAll('button');
                        // Loop backwards to find the one with the specific text
                        for (let i = buttons.length - 1; i >= 0; i--) {{
                            if (buttons[i].innerText.includes("Load Next")) {{
                                buttons[i].click();
                                break;
                            }}
                        }}
                    }}
                }});
            }});
            const sentinel = document.createElement('div');
            sentinel.innerText = "Loading more...";
            sentinel.style.textAlign = "center";
            sentinel.style.padding = "20px";
            sentinel.style.color = "#888";
            document.body.appendChild(sentinel);
            observer.observe(sentinel);
            </script>
            """,
            height=100, # Give it height so it's easily scrollable-to
        )

