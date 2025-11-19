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
    """
    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")
    agg = spec.get("aggregation", "none")

    # Ensure numeric
    df[y] = pd.to_numeric(df[y], errors="coerce")

    # If no aggregation needed → return df
    if agg == "none":
        return df

    # Group columns
    group_cols = [x]
    if group_by:
        group_cols.append(group_by)

    # Aggregation logic
    if agg == "sum":
        agg_df = df.groupby(group_cols, dropna=False)[y].sum().reset_index()
    elif agg == "mean":
        agg_df = df.groupby(group_cols, dropna=False)[y].mean().reset_index()
    elif agg == "count":
        agg_df = df.groupby(group_cols, dropna=False)[y].count().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    return agg_df


def execute_chart_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute chart spec:
    1. Filter df
    2. Aggregate df
    3. Plot chart
    4. Return the aggregated dataframe for debugging / tests
    """
    chart_type = spec["chart_type"]
    x = spec["x"]
    y = spec["y"]
    filters = spec.get("filters", [])

    # Step 1 — filtering
    df_filtered = apply_filters(df, filters)

    # Step 2 — aggregation
    df_agg = aggregate_dataframe(df_filtered, spec)

    # Step 3 — plotting
    plt.figure(figsize=(6, 4))

    if chart_type == "bar":
        plt.bar(df_agg[x], df_agg[y])
        plt.xlabel(x)
        plt.ylabel(f"{spec['aggregation']}({y})" if spec["aggregation"] != "none" else y)
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif chart_type == "line":
        plt.plot(df_agg[x], df_agg[y])
        plt.xlabel(x)
        plt.ylabel(f"{spec['aggregation']}({y})" if spec["aggregation"] != "none" else y)
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif chart_type == "scatter":
        plt.scatter(df_agg[x], df_agg[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif chart_type == "histogram":
        plt.hist(df_agg[y])
        plt.xlabel(y)
        plt.ylabel("count")
        plt.tight_layout()

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    plt.show()

    return df_agg