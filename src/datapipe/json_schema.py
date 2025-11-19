import io
import uuid
from typing import Dict, Any, List
import warnings

import pandas as pd


def infer_dtype(series: pd.Series) -> str:
    """Infer a simple logical dtype string for a pandas Series."""
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # TODO: 后面可以加上 category 等更细类型
    return "string"


def build_columns_meta(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build metadata for each column in the DataFrame."""
    columns_meta: List[Dict[str, Any]] = []

    for col in df.columns:
        s = df[col]
        dtype = infer_dtype(s)

        col_meta: Dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "nullable": bool(s.isna().any()),
            "example_values": s.dropna().astype(str).head(3).tolist(),
        }

        if dtype in {"int", "float"}:
            # 注意 float() 转一下，避免 np 类型不能 JSON 序列化
            col_meta["summary"] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
            }

        columns_meta.append(col_meta)

    return columns_meta


def _try_parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    只对列名看起来像日期/时间的列尝试解析成 datetime。
    避免误伤 price / cost / profit 这种数值列。
    """
    DATE_LIKE_KEYWORDS = ["date", "time", "day", "month", "year", "dt"]

    for col in df.columns:
        col_lower = str(col).lower()

        # 只有列名中包含这些关键字才尝试当成日期
        if not any(k in col_lower for k in DATE_LIKE_KEYWORDS):
            continue

        # 只对 object / string 列尝试解析
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
            except Exception:
                # 解析失败就跳过
                continue
            else:
                df[col] = parsed

    return df


def csv_bytes_to_llm_json(
    file_bytes: bytes,
    dataset_name: str = "uploaded_dataset",
) -> Dict[str, Any]:
    """
    Core function:
    - Input: raw CSV bytes (e.g. from file upload)
    - Output: a standardized JSON structure suitable for LLM usage.
    """
    # 从 bytes 读 CSV
    df = pd.read_csv(io.BytesIO(file_bytes))

    # 尝试把可能是日期的列转成 datetime
    df = _try_parse_datetimes(df)

    dataset_id = str(uuid.uuid4())
    columns_meta = build_columns_meta(df)

    meta: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns_meta,
    }

    data_sample = df.head(5).to_dict(orient="records")

    result: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "source_type": "csv_upload",
        "meta": meta,
        "data_sample": data_sample,
        # 先占位，后面我们再接 storage 层
        "storage": {
            "full_data_location": None,
            "stored_as": None,
        },
    }

    return result