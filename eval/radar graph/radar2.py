import pandas as pd
import plotly.graph_objects as go

# 1. 读入你的打分表（路径按你实际情况改）
df = pd.read_csv("eval/Final_AB_Evaluation_Scores_multi_dataset.csv")

# 2. 维度名字（和 CSV 列名后缀一致）
dimensions = [
    "Question_Clarity",
    "Analytical_Depth",
    "Data_Answerability",
    "Visualization_Appropriateness",
    "Encoding_Correctness",
    "Readability_Visual_Design",
    "QV_Alignment",
    "Overall_Usefulness",
    "Operational_Complexity",
]

labels = [
    "Question Clarity",
    "Analytical Depth",
    "Data Answerability",
    "Viz Appropriateness",
    "Encoding Correctness",
    "Readability & Design",
    "Q–Viz Alignment",
    "Overall Usefulness",
    "Operational Complexity",
]

A_means = []
B_means = []

for d in dimensions:
    A_mean = df[f"A_{d}"].astype(float).mean()
    B_mean = df[f"B_{d}"].astype(float).mean()
    A_means.append(A_mean)
    B_means.append(B_mean)

print("A means:", A_means)
print("B means:", B_means)

# 3. 画雷达图
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=A_means,
    theta=labels,
    fill='toself',
    name='Condition A – Direct LLM'
))

fig.add_trace(go.Scatterpolar(
    r=B_means,
    theta=labels,
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
    title="Radar Comparison of LLM vs. Pipeline (Mean Scores)",
    showlegend=True
)

# 4. 显示 & 导出图片（导出需要安装 kaleido: `pip install kaleido`）
fig.show()
# fig.write_image("eval/radar_evaluation.png")