import plotly.graph_objects as go

# 9 evaluation dimensions (order must match your CSV)
categories = [
    "Question Clarity",
    "Analytical Depth",
    "Data Answerability",
    "Visualization Appropriateness",
    "Encoding Correctness",
    "Readability & Visual Design",
    "Question–Visualization Alignment",
    "Overall Usefulness",
    "Operational Complexity"
]

# ---- REPLACE THESE WITH YOUR REAL MEAN SCORES ----
A_scores = [4.1, 4.5, 4.2, 3.8, 4.0, 3.7, 3.6, 3.9, 2.4]  # Condition A (LLM)
B_scores = [4.0, 4.0, 4.3, 4.4, 4.5, 4.6, 4.5, 4.6, 4.8]  # Condition B (Pipeline)
# --------------------------------------------------

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=A_scores,
    theta=categories,
    fill='toself',
    name='Condition A – Direct LLM'
))

fig.add_trace(go.Scatterpolar(
    r=B_scores,
    theta=categories,
    fill='toself',
    name='Condition B – Pipeline'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 5]
        )
    ),
    title="Radar Comparison of LLM vs. Pipeline",
    showlegend=True
)

fig.show()