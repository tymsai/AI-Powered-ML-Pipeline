from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
import numpy as np
from datetime import datetime
import os
import pandas as pd
import joblib
import json
from typing import List, Optional, Dict, Any

mcp = FastMCP(name="DataPipelineTools")

# -------------------------
# Utility helpers
# -------------------------
def _ensure_base_path(base: str) -> str:
    p = os.path.abspath(base)
    os.makedirs(p, exist_ok=True)
    return p

# ----------------------------
# Helper: Convert Pandas/numpy types to native Python
# ----------------------------
def to_native(obj):
    import numpy as np
    import pandas as pd

    if isinstance(obj, list):
        return [to_native(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(obj)
    else:
        return obj


DATA_ROOT = _ensure_base_path("data")
MODELS_ROOT = _ensure_base_path("models")
ARTIFACTS_ROOT = _ensure_base_path("artifacts")


# --- Tool: List files on local filesystem (in an allowed directory) ---
@mcp.tool(
    name="list_files",
    description="List files in a directory under the data folder"
)
def list_files(dir_path: Optional[str] = None) -> List[str]:
    """
    List files under DATA_ROOT. Prevents escaping root.
    """
    if dir_path:
        target = os.path.abspath(os.path.join(DATA_ROOT, dir_path))
    else:
        target = DATA_ROOT

    if not target.startswith(DATA_ROOT):
        raise ToolError(f"Access denied to directory: {dir_path}")

    if not os.path.exists(target):
        return []
    return os.listdir(target)

@mcp.tool(name="train_test_split")
def train_test_split(dataframe_json, test_size=0.2, random_state=42):
    """Split data before any feature engineering"""
    df = pd.DataFrame(dataframe_json)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return {
        "train_data": train.to_dict(orient="records"),
        "test_data": test.to_dict(orient="records")
    }

# --- Tool: Read a CSV or Excel file into JSON / dict (returns ToolResult) ---
@mcp.tool(
    name="read_file",
    description="Read a CSV or Excel file and return full data with metadata"
)
def read_file(filename: str) -> ToolResult:
    """
    Reads a CSV/Excel file and returns ALL rows as JSON.
    """
    file_path = os.path.abspath(os.path.join(DATA_ROOT, filename))
    if not file_path.startswith(DATA_ROOT):
        raise ToolError("Access to this path is not allowed.")
    if not os.path.exists(file_path):
        raise ToolError(f"File not found: {filename}")

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            raise ToolError("Unsupported file format. Only CSV and Excel allowed.")
    except Exception as e:
        raise ToolError(f"Error reading file: {str(e)}")

    # Return FULL data (not just preview)
    full_data = df.to_dict(orient="records")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Optionally save pickle for downstream tools
    pickle_name = f"{os.path.splitext(filename)[0]}.pkl"
    pickle_path = os.path.join(DATA_ROOT, pickle_name)
    try:
        df.to_pickle(pickle_path)
        saved_pickled = pickle_path
    except Exception:
        saved_pickled = None

    return ToolResult(
        structured_content={
            "data": full_data,        # ← ALL rows (changed from "preview")
            "dtypes": dtypes,
            "num_rows": int(len(df)),
            "num_cols": int(len(df.columns)),
            "pickle_path": saved_pickled
        }
    )


# --- Tool: Save a DataFrame (as pickle) on local filesystem ---
@mcp.tool(
    name="save_dataframe",
    description="Save a dataframe (sent as JSON records) to local disk for later steps"
)
def save_dataframe(
    filename: str,
    dataframe_json: List[Dict[str, Any]]
) -> str:
    save_path = os.path.abspath(os.path.join(DATA_ROOT, filename))
    if not save_path.startswith(DATA_ROOT):
        raise ToolError("Invalid save path.")

    try:
        df = pd.DataFrame(dataframe_json)
    except Exception as e:
        raise ToolError(f"Unable to convert JSON to DataFrame: {e}")

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_pickle(save_path)
    except Exception as e:
        raise ToolError(f"Failed to save DataFrame: {e}")

    return save_path


# --- Tool: Dynamic data cleaning (returns ToolResult) ---
@mcp.tool(
    name="data_cleaning",
    description="Apply dynamic cleaning operations on a dataset according to instructions from LLM"
)
def data_cleaning(
    dataframe_json: List[Dict[str, Any]],
    cleaning_plan: Dict[str, Any]
) -> ToolResult:

    df = pd.DataFrame(dataframe_json)

    # 1) Drop columns
    for col in cleaning_plan.get("drop_columns", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2) Drop columns with missing ratio above threshold
    threshold = cleaning_plan.get("drop_missing_threshold", None)
    if threshold is not None:
        try:
            threshold = float(threshold)
            cols_to_keep = [c for c in df.columns if df[c].isna().mean() <= threshold]
            df = df[cols_to_keep]
        except Exception:
            pass

    # 3) Fill numeric missing values
    for col, val in cleaning_plan.get("fillna_numeric", {}).items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    # 4) Fill categorical missing values
    for col, val in cleaning_plan.get("fillna_categorical", {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # 5) Convert to datetime if requested
    for item in cleaning_plan.get("convert_datetime", []):
        if isinstance(item, dict):
            col = item.get("col")
            fmt = item.get("format")
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                except Exception:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

    # 6) Drop duplicates
    if cleaning_plan.get("drop_duplicates", False):
        df = df.drop_duplicates()

    # --- SANITIZE EVERYTHING FOR JSON ---
    df_json_safe = df.applymap(
        lambda x:
            x.isoformat() if isinstance(x, (pd.Timestamp, datetime))
            else float(x) if isinstance(x, (np.floating,))
            else int(x) if isinstance(x, (np.integer,))
            else None if pd.isna(x)
            else x
    ).to_dict(orient="records")

    # Return JSON-safe result
    return ToolResult(
        structured_content={
            "cleaned_data": df_json_safe,
            "num_rows": int(df.shape[0]),
            "num_cols": int(df.shape[1])
        }
    )

# --- Tool: Load a DataFrame (pickle) ---
@mcp.tool(
    name="load_dataframe",
    description="Load a previously saved dataframe pickle"
)
def load_dataframe(pickle_path: str) -> dict:
    full_path = os.path.abspath(os.path.join(DATA_ROOT, pickle_path))
    if not full_path.startswith(DATA_ROOT):
        raise ToolError("Access denied.")
    if not os.path.exists(full_path):
        raise ToolError(f"Pickle file not found: {pickle_path}")
    df = pd.read_pickle(full_path)
    preview = df.head(10).to_dict(orient="records")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return {"preview": preview, "dtypes": dtypes, "num_rows": int(len(df)), "num_cols": int(len(df.columns))}


# --- Tool: Train a model given parameters ---
@mcp.tool(
    name="train_model",
    description="Train a scikit-learn or xgboost model with given parameters (auto-detect regression/classification) and return feature info"
)
def train_model(
    model_type: str,
    params: dict,
    train_data: List[Dict[str, Any]],
    target_column: str
) -> dict:
    """
    Train a model given training data (as JSON records) + target column.
    Supports:
        - LogisticRegression, RandomForest, DecisionTree, XGBoost
        - Auto-selects classifier or regressor based on target type
    Returns: model path, type, task type, column names, and feature importances (if applicable)
    """
    import os
    import pandas as pd
    import joblib
    from fastmcp.exceptions import ToolError
    import numpy as np

    # Convert training data JSON to DataFrame
    try:
        df = pd.DataFrame(train_data)
    except Exception as e:
        raise ToolError(f"Invalid training data: {e}")

    if target_column not in df.columns:
        raise ToolError(f"target_column {target_column} not in data")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Detect if target is regression or classification
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        task_type = "regression"
    else:
        task_type = "classification"

    # Import models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        XGBClassifier = XGBRegressor = None

    model = None
    model_filename = f"model_{model_type}_{task_type}.pkl"
    model_path = os.path.abspath(os.path.join("models", model_filename))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Choose model based on type and task
    if model_type == "logistic_regression":
        if task_type != "classification":
            raise ToolError("LogisticRegression only supports classification")
        model = LogisticRegression(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params) if task_type == "classification" else RandomForestRegressor(**params)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(**params) if task_type == "classification" else DecisionTreeRegressor(**params)
    elif model_type == "xgboost":
        if XGBClassifier is None or XGBRegressor is None:
            raise ToolError("XGBoost is not installed.")
        model = XGBClassifier(**params) if task_type == "classification" else XGBRegressor(**params)
    else:
        raise ToolError(f"Unsupported model_type: {model_type}")

    # Train model
    try:
        model.fit(X, y)
    except Exception as e:
        raise ToolError(f"Training failed: {e}")

    # Save model
    joblib.dump(model, model_path)

    # Extract feature importances if available
    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = dict(zip(X.columns, model.feature_importances_.tolist()))
    elif hasattr(model, "coef_"):  # For linear models
        feature_importances = dict(zip(X.columns, model.coef_.ravel().tolist()))

    return {
        "model_path": model_path,
        "model_type": model_type,
        "task_type": task_type,
        "column_names": list(X.columns),
        "feature_importances": feature_importances
    }



# --- Tool: Evaluate a saved model on test data ---
@mcp.tool(
    name="evaluate_model",
    description="Evaluate a trained model on test data and return metrics"
)
def evaluate_model(
    model_path: str,
    test_data: List[Dict[str, Any]],
    target_column: str,
    problem_type: str = "classification"
) -> dict:
    full_model_path = os.path.abspath(model_path)
    if not os.path.exists(full_model_path):
        raise ToolError(f"Model not found at path: {model_path}")

    model = joblib.load(full_model_path)

    try:
        df = pd.DataFrame(test_data)
    except Exception as e:
        raise ToolError(f"Invalid test data: {e}")

    if target_column not in df.columns:
        raise ToolError(f"target_column {target_column} not in test data")

    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    result: Dict[str, Any] = {}
    if problem_type == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = model.predict(X_test)
        result["accuracy"] = float(accuracy_score(y_test, y_pred))
        result["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        result["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        result["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_pred = model.predict(X_test)
        result["mse"] = float(mean_squared_error(y_test, y_pred))
        result["mae"] = float(mean_absolute_error(y_test, y_pred))
        result["r2"] = float(r2_score(y_test, y_pred))

    return result


# --- Tool: Delete a file (cleanup) ---
@mcp.tool(
    name="delete_file",
    description="Delete a file from the data or models folder"
)
def delete_file(path: str) -> dict:
    target = os.path.abspath(path)
    cwd = os.path.abspath(".")
    if not target.startswith(cwd):
        raise ToolError("Invalid path for deletion.")
    try:
        os.remove(target)
        return {"deleted": True, "path": path}
    except Exception as e:
        raise ToolError(f"Could not delete file: {e}")

# --- Tool: Automatic schema extraction (converted to ToolResult earlier) ---
@mcp.tool(
    name="extract_schema",
    description="Automatically extract dynamic schema, statistics and column metadata from a dataset"
)
def extract_schema(dataframe_json: List[Dict[str, Any]]) -> ToolResult:
    df = pd.DataFrame(dataframe_json)
    schema_info = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            suggested = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            suggested = "datetime"
        else:
            suggested = "categorical"

        col_info: Dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "missing_pct": float(series.isna().mean()),
            "unique_values": int(series.nunique()),
            "suggested_type": suggested
        }

        if pd.api.types.is_numeric_dtype(series):
            # safe conversions: if no valid values, return None
            if series.count() > 0:
                try:
                    col_info["stats"] = {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "std": float(series.std())
                    }
                except Exception:
                    col_info["stats"] = {}
            else:
                col_info["stats"] = {}
        else:
            # sample up to 10 unique non-null values for categorical to help LLM context
            try:
                sample_vals = series.dropna().unique()[:10].tolist()
                col_info["sample_values"] = sample_vals
            except Exception:
                col_info["sample_values"] = []

        schema_info.append(col_info)

    return ToolResult(
        structured_content={
            "num_rows": int(len(df)),
            "num_cols": int(len(df.columns)),
            "columns": schema_info
        }
    )


# --- Tool: Save arbitrary JSON artifact (like metrics or report) ---
@mcp.tool(
    name="save_json",
    description="Save JSON content to a file (e.g. evaluation metrics or report)"
)
def save_json(filename: str, content: Dict[str, Any]) -> str:
    os.makedirs(ARTIFACTS_ROOT, exist_ok=True)
    path = os.path.abspath(os.path.join(ARTIFACTS_ROOT, filename))
    with open(path, "w") as f:
        json.dump(content, f, indent=2)
    return path

@mcp.tool(
    name="feature_engineering",
    description="Apply dynamic feature engineering transformations according to instructions from LLM; ensures all features are numeric for ML."
)
def feature_engineering(
    dataframe_json: List[Dict[str, Any]],
    fe_plan: Dict[str, Any]
) -> ToolResult:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures

    df = pd.DataFrame(dataframe_json)

    # 0) Drop unwanted columns
    for col in fe_plan.get("drop_columns", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    # 1) Convert datetime columns to numeric features
    for item in fe_plan.get("datetime_extract", []):
        if isinstance(item, dict):
            col = item.get("col")
            parts = item.get("parts", ["year", "month", "day"])
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                for p in parts:
                    if p == "year":
                        df[f"{col}_year"] = df[col].dt.year
                    elif p == "month":
                        df[f"{col}_month"] = df[col].dt.month
                    elif p == "day":
                        df[f"{col}_day"] = df[col].dt.day
                    elif p == "hour":
                        df[f"{col}_hour"] = df[col].dt.hour
                    elif p == "weekday":
                        df[f"{col}_weekday"] = df[col].dt.weekday
                df = df.drop(columns=[col])

    # 2) One-hot encode categorical columns
    # FIXED: Handle categorical_mappings for consistent encoding
    one_hot_cols = fe_plan.get("one_hot_columns", [])
    categorical_mappings = fe_plan.get("categorical_mappings", {})
    
    for col in one_hot_cols:
        if col in df.columns:
            # If mappings provided (from training), use them to ensure consistency
            if col in categorical_mappings:
                expected_categories = categorical_mappings[col]
                # Create dummy columns for ALL expected categories
                for cat in expected_categories:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df = df.drop(columns=[col])
            else:
                # Training phase: create dummies normally
                if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
                    df = pd.get_dummies(df, columns=[col], drop_first=True)

    # 3) Scale numeric columns
    numeric_cols = [col for col in fe_plan.get("scale_numeric", []) if col in df.columns]
    if numeric_cols:
        scaler_params = fe_plan.get("scaler_params", {})
        
        if scaler_params:
            # Use pre-fitted scaler parameters (from training)
            for col in numeric_cols:
                if col in scaler_params:
                    mean = scaler_params[col]["mean"]
                    std = scaler_params[col]["std"]
                    df[col] = (df[col] - mean) / std if std > 0 else df[col] - mean
        else:
            # Fit new scaler (training phase)
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 4) Polynomial features
    poly_cols = fe_plan.get("polynomial_features", [])
    if poly_cols and isinstance(poly_cols, list):
        valid_cols = [col for col in poly_cols if col in df.columns]
        if valid_cols:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_array = poly.fit_transform(df[valid_cols])
            poly_df = pd.DataFrame(
                poly_array,
                columns=poly.get_feature_names_out(valid_cols),
                index=df.index
            )
            df = df.drop(columns=valid_cols)
            df = pd.concat([df, poly_df], axis=1)

    # 5) Force all remaining object columns to numeric (coerce errors)
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Fill any NaN created by coercion
    df = df.fillna(0)

    return ToolResult(
        structured_content={
            "fe_preview": df.head(10).to_dict(orient="records"),
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "column_names": list(df.columns)  # Return column names for verification
        }
    )


# ============================================================
# NEW TOOL: fit_feature_engineering (for training phase)
# ============================================================
# Add this to your server.py - REPLACE the existing fit_feature_engineering tool

@mcp.tool(
    name="fit_feature_engineering",
    description="Fit feature engineering on training data and return parameters for test data transformation"
)
def fit_feature_engineering(
    dataframe_json: List[Dict[str, Any]],
    fe_plan: Dict[str, Any]
) -> ToolResult:
    """
    Fits feature engineering transformations and returns:
    1. Transformed training data (WITH target column preserved)
    2. Parameters needed to transform test data identically
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures

    df = pd.DataFrame(dataframe_json)

    # Track transformation parameters
    transform_params = {
        "categorical_mappings": {},
        "scaler_params": {},
        "column_order": [],
        "target_column": None  # Track which column is the target
    }

    # CRITICAL: Extract and save target column BEFORE any transformations
    target_col = None
    for col in fe_plan.get("drop_columns", []):
        if col in df.columns:
            # This is likely the target - save it separately
            target_col = col
            target_series = df[col].copy()
            transform_params["target_column"] = col
            df = df.drop(columns=[col])

    # 0) Drop other unwanted columns (non-target)
    for col in fe_plan.get("drop_columns", []):
        if col in df.columns and col != target_col:
            df = df.drop(columns=[col])

    # 1) Convert datetime columns to numeric features
    for item in fe_plan.get("datetime_extract", []):
        if isinstance(item, dict):
            col = item.get("col")
            parts = item.get("parts", ["year", "month", "day"])
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                for p in parts:
                    if p == "year":
                        df[f"{col}_year"] = df[col].dt.year
                    elif p == "month":
                        df[f"{col}_month"] = df[col].dt.month
                    elif p == "day":
                        df[f"{col}_day"] = df[col].dt.day
                    elif p == "hour":
                        df[f"{col}_hour"] = df[col].dt.hour
                    elif p == "weekday":
                        df[f"{col}_weekday"] = df[col].dt.weekday
                df = df.drop(columns=[col])

    # 2) One-hot encode and save category mappings
    for col in fe_plan.get("one_hot_columns", []):
        if col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
                # Save unique categories for this column
                unique_cats = df[col].dropna().unique().tolist()
                transform_params["categorical_mappings"][col] = unique_cats
                
                # Create dummies
                df = pd.get_dummies(df, columns=[col], drop_first=True)

    # 3) Scale numeric columns and save parameters
    numeric_cols = [col for col in fe_plan.get("scale_numeric", []) if col in df.columns]
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Save mean and std for each column
        for i, col in enumerate(numeric_cols):
            transform_params["scaler_params"][col] = {
                "mean": float(scaler.mean_[i]),
                "std": float(scaler.scale_[i])
            }

    # 4) Polynomial features
    poly_cols = fe_plan.get("polynomial_features", [])
    if poly_cols and isinstance(poly_cols, list):
        valid_cols = [col for col in poly_cols if col in df.columns]
        if valid_cols:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_array = poly.fit_transform(df[valid_cols])
            poly_df = pd.DataFrame(
                poly_array,
                columns=poly.get_feature_names_out(valid_cols),
                index=df.index
            )
            df = df.drop(columns=valid_cols)
            df = pd.concat([df, poly_df], axis=1)

    # 5) Force all remaining object columns to numeric
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.fillna(0)

    # Save final column order (WITHOUT target)
    transform_params["column_order"] = list(df.columns)

    # CRITICAL: Re-add the target column at the END
    if target_col is not None:
        df[target_col] = target_series

    return ToolResult(
        structured_content={
            "fe_data": df.to_dict(orient="records"),
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "column_names": list(df.drop(columns=[target_col]).columns) if target_col else list(df.columns),
            "transform_params": transform_params
        }
    )


# ============================================================
# NEW TOOL: transform_feature_engineering (for test phase)
# ============================================================
@mcp.tool(
    name="transform_feature_engineering",
    description="Transform test data using parameters fitted on training data"
)
def transform_feature_engineering(
    dataframe_json: List[Dict[str, Any]],
    fe_plan: Dict[str, Any],
    transform_params: Dict[str, Any]
) -> ToolResult:
    """
    Apply feature engineering to test data using pre-fitted parameters
    """
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures

    df = pd.DataFrame(dataframe_json)

    # CRITICAL: Extract target column before transformations
    target_col = transform_params.get("target_column")
    target_series = None
    if target_col and target_col in df.columns:
        target_series = df[target_col].copy()
        df = df.drop(columns=[target_col])

    # 0) Drop unwanted columns (except target)
    for col in fe_plan.get("drop_columns", []):
        if col in df.columns and col != target_col:
            df = df.drop(columns=[col])

    # 1) Convert datetime columns
    for item in fe_plan.get("datetime_extract", []):
        if isinstance(item, dict):
            col = item.get("col")
            parts = item.get("parts", ["year", "month", "day"])
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                for p in parts:
                    if p == "year":
                        df[f"{col}_year"] = df[col].dt.year
                    elif p == "month":
                        df[f"{col}_month"] = df[col].dt.month
                    elif p == "day":
                        df[f"{col}_day"] = df[col].dt.day
                    elif p == "hour":
                        df[f"{col}_hour"] = df[col].dt.hour
                    elif p == "weekday":
                        df[f"{col}_weekday"] = df[col].dt.weekday
                df = df.drop(columns=[col])

    # 2) One-hot encode using saved category mappings
    categorical_mappings = transform_params.get("categorical_mappings", {})
    for col in fe_plan.get("one_hot_columns", []):
        if col in df.columns and col in categorical_mappings:
            expected_categories = categorical_mappings[col]
            # Create dummy for ALL categories seen in training
            for cat in expected_categories:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df = df.drop(columns=[col])

    # 3) Scale using saved parameters
    scaler_params = transform_params.get("scaler_params", {})
    numeric_cols = [col for col in fe_plan.get("scale_numeric", []) if col in df.columns]
    for col in numeric_cols:
        if col in scaler_params:
            mean = scaler_params[col]["mean"]
            std = scaler_params[col]["std"]
            df[col] = (df[col] - mean) / std if std > 0 else df[col] - mean

    # 4) Polynomial features
    poly_cols = fe_plan.get("polynomial_features", [])
    if poly_cols and isinstance(poly_cols, list):
        valid_cols = [col for col in poly_cols if col in df.columns]
        if valid_cols:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_array = poly.fit_transform(df[valid_cols])
            poly_df = pd.DataFrame(
                poly_array,
                columns=poly.get_feature_names_out(valid_cols),
                index=df.index
            )
            df = df.drop(columns=valid_cols)
            df = pd.concat([df, poly_df], axis=1)

    # 5) Force remaining objects to numeric
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.fillna(0)

    # 6) Ensure column order matches training data
    expected_columns = transform_params.get("column_order", [])
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]

    # CRITICAL: Re-add target column at the END
    if target_col and target_series is not None:
        df[target_col] = target_series

    return ToolResult(
        structured_content={
            "fe_data": df.to_dict(orient="records"),
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "column_names": expected_columns  # Return features WITHOUT target
        }
    )


# If run as script → start server
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(MODELS_ROOT, exist_ok=True)
    os.makedirs(ARTIFACTS_ROOT, exist_ok=True)

    # Start the FastMCP server with default transport
    mcp.run()
