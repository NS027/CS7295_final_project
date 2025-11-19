from pathlib import Path
import pandas as pd
from datapipe.viz_executor import execute_chart_spec

def test_execute_chart_spec_mean_price_by_category():
    df = pd.read_csv(Path("data/sample_orders.csv"))
    spec = {
        "chart_type": "bar",
        "x": "category",
        "y": "price",
        "aggregation": "mean",
        "group_by": None,
        "filters": []
    }
    agg_df = execute_chart_spec(df, spec)
    assert set(agg_df["category"]) == {"Clothing", "Electronics", "Furniture"}
    assert "price" in agg_df.columns