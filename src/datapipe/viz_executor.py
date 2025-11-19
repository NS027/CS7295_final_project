import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any


def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    """Apply simple filter operations defined in chart spec."""
    filtered_df = df.copy()

    for f in filters:
        col = f["column"]
        op = f["op"]
        value = f["value"]

        if op == "==":
            filtered_df = filtered_df[filtered_df[col] == value]
        elif op == "!=":
            filtered_df = filtered_df[filtered_df[col] != value]
        elif op == ">":
            filtered_df = filtered_df[filtered_df[col] > value]
        elif op == "<":
            filtered_df = filtered_df[filtered_df[col] < value]
        elif op == ">=":
            filtered_df = filtered_df[filtered_df[col] >= value]
        elif op == "<=":
            filtered_df = filtered_df[filtered_df[col] <= value]

    return filtered_df


def aggregate_dataframe(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregate dataframe according to the chart spec.
    Supports: sum, mean, count, none.

    - 自动把 y 列转换成数值
    - 自动避免 group_by 与 x 重复
    - 聚合后列名统一为 "<agg>_<y>"
    """
    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")
    agg = spec.get("aggregation", "none")

    # 确保 y 列是数值
    df = df.copy()
    df[y] = pd.to_numeric(df[y], errors="coerce")

    # 不需要聚合就直接返回
    if agg == "none":
        return df

    # 构造分组列，去重
    group_cols = [x]
    if group_by and group_by not in group_cols:
        group_cols.append(group_by)

    # 选择聚合函数
    if agg == "sum":
        agg_func = "sum"
    elif agg == "mean":
        agg_func = "mean"
    elif agg == "count":
        agg_func = "count"
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    # 聚合并且直接指定新列名，避免 reset_index 的列名冲突
    new_col_name = f"{agg}_{y}"
    agg_df = (
        df.groupby(group_cols, dropna=False)[y]
        .agg(agg_func)
        .reset_index(name=new_col_name)
    )

    return agg_df


def execute_chart_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute chart spec:
    1. Filter df
    2. Aggregate df
    3. Plot chart
    4. Return the aggregated dataframe
    """
    chart_type = spec.get("chart_type")
    x = spec.get("x")
    y = spec.get("y")
    filters = spec.get("filters", [])
    agg = spec.get("aggregation", "none")

    # 1. 过滤
    df_filtered = apply_filters(df, filters)

    # 2. 聚合
    df_agg = aggregate_dataframe(df_filtered, spec)

    # 3. 确定要画的数值列名
    #   - 如果有聚合: mean_profit / sum_price / count_order_id ...
    #   - 没有聚合: 就是原始 y
    if agg != "none":
        value_col = f"{agg}_{y}"
    else:
        value_col = y

    if value_col not in df_agg.columns:
        # 防御一下，方便调试
        raise ValueError(
            f"value_col '{value_col}' not in aggregated dataframe columns: {df_agg.columns.tolist()}"
        )

    # 4. 画图
    plt.figure(figsize=(6, 4))

    if chart_type == "bar":
        plt.bar(df_agg[x], df_agg[value_col])
        plt.xlabel(x)
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif chart_type == "line":
        plt.plot(df_agg[x], df_agg[value_col])
        plt.xlabel(x)
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif chart_type == "scatter":
        plt.scatter(df_agg[x], df_agg[value_col])
        plt.xlabel(x)
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif chart_type == "histogram":
        plt.hist(df_agg[value_col])
        plt.xlabel(value_col)
        plt.ylabel("count")
        plt.tight_layout()

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    plt.show()  # API 环境下不弹窗，避免阻塞

    return df_agg