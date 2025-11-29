import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

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

def render_chart(df_agg, spec):
    """
    Render chart based on spec with support for multiple chart types
    """
    chart_type = spec.get("chart_type", "bar")
    x = spec.get("x")
    y = spec.get("y")
    agg = spec.get("aggregation", "none")
    color = spec.get("color")
    
    # Determine value column name
    if agg != "none":
        value_col = f"{agg}_{y}"
    else:
        value_col = y
    
    # Check if value column exists
    if value_col not in df_agg.columns and y in df_agg.columns:
        value_col = y
    
    try:
        # Bar Chart
        if chart_type == "bar":
            fig = px.bar(df_agg, x=x, y=value_col, color=color, 
                        title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Line Chart
        elif chart_type == "line":
            fig = px.line(df_agg, x=x, y=value_col, color=color,
                         title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter Plot
        elif chart_type == "scatter":
            size_col = spec.get("size")
            fig = px.scatter(df_agg, x=x, y=value_col, color=color, size=size_col,
                           title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Histogram
        elif chart_type == "histogram":
            fig = px.histogram(df_agg, x=value_col, nbins=30,
                             title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Pie Chart
        elif chart_type == "pie":
            fig = px.pie(df_agg, names=x, values=value_col,
                        title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Stacked Bar Chart
        elif chart_type == "stacked_bar":
            group_by = spec.get("group_by") or color
            if group_by:
                fig = px.bar(df_agg, x=x, y=value_col, color=group_by,
                           title=spec.get("title", ""), height=350,
                           barmode='stack')
            else:
                # Fallback to regular bar if no grouping
                fig = px.bar(df_agg, x=x, y=value_col,
                           title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Grouped Bar Chart
        # Grouped Bar Chart
        elif chart_type == "grouped_bar":
            group_by = spec.get("group_by") or color
            if group_by and group_by in df_agg.columns:
                # Make sure we have aggregated data
                fig = px.bar(df_agg, x=x, y=value_col, color=group_by,
                        title=spec.get("title", ""), height=350,
                        barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to regular bar if no grouping column
                st.warning("Grouped bar chart requires a grouping column. Showing regular bar chart.")
                fig = px.bar(df_agg, x=x, y=value_col,
                        title=spec.get("title", ""), height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Box Plot
        elif chart_type == "box":
            fig = px.box(df_agg, x=x, y=value_col, color=color,
                        title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Area Chart
        elif chart_type == "area":
            fig = px.area(df_agg, x=x, y=value_col, color=color,
                         title=spec.get("title", ""), height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Fallback to simple streamlit charts
        else:
            st.warning(f"Chart type '{chart_type}' not fully supported, using bar chart")
            st.bar_chart(df_agg.set_index(x)[value_col], height=300)
            
    except Exception as e:
        st.error(f"Chart rendering error: {e}")
        st.write("Debug info:")
        st.write(f"Chart type: {chart_type}")
        st.write(f"Columns available: {df_agg.columns.tolist()}")
        st.dataframe(df_agg.head())

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
                    
                    # Add this check before rendering
                    if df_agg is None or df_agg.empty:
                        st.warning("⚠️ No data returned for this query")
                    else:
                        # Use the new render_chart function
                        render_chart(df_agg, spec)
                    
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

        # Chart type diversity stats in sidebar
    if st.session_state["chart_specs_map"]:
        chart_types = [s.get("chart_type", "unknown") 
                      for s in st.session_state["chart_specs_map"].values() 
                      if s is not None]
        if chart_types:
            type_counts = pd.Series(chart_types).value_counts()
            with st.sidebar:
                st.subheader("📊 Chart Type Distribution")
                st.bar_chart(type_counts)
        
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

