import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    """Apply simple filter operations defined in chart spec."""
    filtered_df = df.copy()

    for f in filters:
        col = f.get("column")
        op = f.get("op")
        value = f.get("value")

        # Validate column exists
        if col not in filtered_df.columns:
            print(f"Warning: Filter column '{col}' not found, skipping filter")
            continue

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


def aggregate_dataframe(df: pd.DataFrame, spec: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Aggregate dataframe according to the chart spec.
    Supports: sum, mean, count, max, min, median, none.

    - Converts y to numeric
    - Avoids duplicated group_by columns
    - Names aggregated column as "<agg>_<y>"
    """
    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")
    agg = spec.get("aggregation", "none")

    # Defensive: normalize to lowercase string
    if isinstance(agg, str):
        agg = agg.lower()
    else:
        agg = "none"
    
    # Validate required columns exist
    if x not in df.columns:
        print(f"Error: x column '{x}' not found in dataframe")
        return None
    
    if y not in df.columns:
        print(f"Error: y column '{y}' not found in dataframe")
        return None

    # Work on a copy and ensure y is numeric
    df = df.copy()
    df[y] = pd.to_numeric(df[y], errors="coerce")

    # No aggregation -> return original (filtered) df
    if agg in ("none", "", None):
        return df
    
    # For count aggregation, we don't need y to be numeric
    if agg != "count":
        # Convert y to numeric for other aggregations
        df[y] = pd.to_numeric(df[y], errors="coerce")
        # Drop rows where y couldn't be converted
        df = df.dropna(subset=[y])
        
        if df.empty:
            print(f"Error: No valid numeric data in column '{y}'")
            return None

    # Build group columns (deduplicated)
    group_cols = [x]
    if group_by and group_by in df.columns and group_by not in group_cols:
        group_cols.append(group_by)

    # Map supported aggregations
    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "average": "mean",
        "count": "count",
        "max": "max",
        "min": "min",
        "median": "median",
    }

    if agg not in agg_map:
        # Fallback: be forgiving and default to mean
        agg_func = "mean"
        agg = "mean"
    else:
        agg_func = agg_map[agg]

    new_col_name = f"{agg}_{y}"

    try:
        agg_df = (
            df.groupby(group_cols, dropna=False)[y]
            .agg(agg_func)
            .reset_index(name=new_col_name)
        )
        return agg_df
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return None


def execute_chart_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> Optional[pd.DataFrame]:
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

    # 1. Filter
    df_filtered = apply_filters(df, filters)
    
    if df_filtered.empty:
        print("Warning: No data after filtering")
        return None

    # 2. Aggregate
    df_agg = aggregate_dataframe(df_filtered, spec)
    
    if df_agg is None or df_agg.empty:
        print("Warning: No data after aggregation")
        return None

    # 3. Determine value column name
    if agg and agg.lower() not in ("none", "", None):
        value_col = f"{agg}_{y}"
    else:
        value_col = y

    if value_col not in df_agg.columns:
        print(f"Error: value_col '{value_col}' not in aggregated dataframe columns: {df_agg.columns.tolist()}")
        return None

    # 4. Plot (optional - mostly for testing)
    try:
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
            plt.hist(df_agg[value_col].dropna())
            plt.xlabel(value_col)
            plt.ylabel("count")
            plt.tight_layout()

        plt.close()  # Don't show in Streamlit environment
    except Exception as e:
        print(f"Warning: Could not plot chart: {e}")

    return df_agg