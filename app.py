import streamlit as st
import asyncio
import json
import os
import pandas as pd
import ollama
from fastmcp import Client
from server import mcp
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END

# ======================================
# STATE DEFINITION
# ======================================

class MLPipelineState(TypedDict):
    # Input
    filename: str
    target_column: str
    
    # Data at various stages
    raw_data: Optional[List[Dict[str, Any]]]
    schema: Optional[Dict[str, Any]]
    cleaning_plan: Optional[Dict[str, Any]]
    cleaned_data: Optional[List[Dict[str, Any]]]
    train_data: Optional[List[Dict[str, Any]]]
    test_data: Optional[List[Dict[str, Any]]]
    
    # Feature Engineering
    fe_plan: Optional[Dict[str, Any]]
    train_fe_data: Optional[List[Dict[str, Any]]]
    test_fe_data: Optional[List[Dict[str, Any]]]
    transform_params: Optional[Dict[str, Any]]
    
    # Model
    model_choice: Optional[Dict[str, Any]]
    model_path: Optional[str]
    metrics: Optional[Dict[str, Any]]
    
    # Metadata
    task_type: Optional[str]
    column_names: Optional[List[str]]
    num_cols: Optional[int]
    num_rows_train: Optional[int]
    num_rows_test: Optional[int]
    feature_importances: Optional[Dict[str, Any]]
    
    # Retry & control flow
    retry_count: int
    max_retries: int
    needs_retry: bool
    skip_cleaning: bool
    accuracy_threshold: float
    
    # Error tracking
    errors: List[str]
    warnings: List[str]
    
    # UI state tracking
    current_step: str

# ======================================
# HELPER FUNCTIONS
# ======================================

async def get_tool_output(result):
    """Extract output from MCP result"""
    def unwrap(obj):
        if isinstance(obj, list):
            unwrapped = [unwrap(o) for o in obj]
            return unwrapped[0] if len(unwrapped) == 1 else unwrapped
        elif hasattr(obj, "text"):
            try:
                return json.loads(obj.text)
            except:
                return obj.text
        elif isinstance(obj, dict):
            return {k: unwrap(v) for k, v in obj.items()}
        return obj
    return unwrap(result.content)

def ask_ollama(prompt: str) -> dict:
    """Ask Ollama for decision"""
    try:
        response = ollama.chat(
            model='qwen2.5-coder:7b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        st.error(f"Ollama error: {str(e)}")
        return {}

# ======================================
# NODE FUNCTIONS
# ======================================

async def load_data_node(state: MLPipelineState) -> MLPipelineState:
    """Node 1: Load data from file"""
    state["current_step"] = "Loading data"
    
    async with Client(mcp) as client:
        result = await client.call_tool("read_file", {"filename": state["filename"]})
        output = await get_tool_output(result)
        
        state["raw_data"] = output["data"]
        state["num_rows_train"] = output["num_rows"]
        state["num_cols"] = output["num_cols"]
    
    return state

async def extract_schema_node(state: MLPipelineState) -> MLPipelineState:
    """Node 2: Extract schema"""
    state["current_step"] = "Analyzing schema"
    
    async with Client(mcp) as client:
        result = await client.call_tool("extract_schema", {
            "dataframe_json": state["raw_data"]
        })
        state["schema"] = await get_tool_output(result)
    
    return state

async def assess_data_quality_node(state: MLPipelineState) -> MLPipelineState:
    """Node 3: LLM assesses if data needs cleaning"""
    state["current_step"] = "Assessing data quality"
    
    prompt = f"""
Analyze this data schema and determine if cleaning is needed:

{json.dumps(state["schema"], indent=2)}

Consider:
- Missing value percentages
- Data types
- Column quality

Respond with JSON:
{{
  "needs_cleaning": true/false,
  "reasoning": "why or why not",
  "severity": "low/medium/high"
}}
"""
    
    assessment = ask_ollama(prompt)
    
    if not assessment or assessment.get("needs_cleaning", True):
        state["skip_cleaning"] = False
    else:
        state["skip_cleaning"] = True
        state["warnings"].append("Skipping cleaning: data quality is good")
    
    return state

async def llm_plan_cleaning_node(state: MLPipelineState) -> MLPipelineState:
    """Node 4: Ollama plans data cleaning"""
    state["current_step"] = "Planning data cleaning"
    
    prompt = f"""
Analyze this schema and create a cleaning plan:

{json.dumps(state["schema"], indent=2)}

Target: {state["target_column"]}

Rules:
1. Drop columns with >70% missing (NEVER drop target: {state["target_column"]})
2. Fill numeric columns with median
3. Fill categorical with "Unknown"
4. Drop duplicates

JSON format:
{{
  "drop_columns": ["col1"],
  "fillna_numeric": {{"age": 0}},
  "fillna_categorical": {{"category": "Unknown"}},
  "drop_duplicates": true,
  "reasoning": "explanation"
}}
"""
    
    cleaning_plan = ask_ollama(prompt)
    
    if not cleaning_plan:
        state["errors"].append("Failed to get cleaning plan")
        cleaning_plan = {
            "drop_columns": [],
            "fillna_numeric": {},
            "fillna_categorical": {},
            "drop_duplicates": True,
            "reasoning": "Default fallback"
        }
    
    # Safety: don't drop target
    if state["target_column"] in cleaning_plan.get("drop_columns", []):
        cleaning_plan["drop_columns"].remove(state["target_column"])
    
    state["cleaning_plan"] = cleaning_plan
    return state

async def clean_data_node(state: MLPipelineState) -> MLPipelineState:
    """Node 5: Execute data cleaning"""
    state["current_step"] = "Cleaning data"
    
    async with Client(mcp) as client:
        result = await client.call_tool("data_cleaning", {
            "dataframe_json": state["raw_data"],
            "cleaning_plan": state["cleaning_plan"]
        })
        output = await get_tool_output(result)
        
        state["cleaned_data"] = output["cleaned_data"]
        state["num_rows_train"] = output["num_rows"]
        state["num_cols"] = output["num_cols"]
    
    return state

async def split_data_node(state: MLPipelineState) -> MLPipelineState:
    """Node 6: Train/test split"""
    state["current_step"] = "Splitting data"
    
    # Use cleaned data if available, else raw
    data_to_split = state["cleaned_data"] if state["cleaned_data"] else state["raw_data"]
    
    async with Client(mcp) as client:
        result = await client.call_tool("train_test_split", {
            "dataframe_json": data_to_split,
            "test_size": 0.2,
            "random_state": 42
        })
        output = await get_tool_output(result)
        
        state["train_data"] = output["train_data"]
        state["test_data"] = output["test_data"]
        state["num_rows_train"] = len(output["train_data"])
        state["num_rows_test"] = len(output["test_data"])
    
    return state

async def llm_plan_fe_node(state: MLPipelineState) -> MLPipelineState:
    """Node 7: Ollama plans feature engineering"""
    state["current_step"] = "Planning feature engineering"
    
    data_source = state["cleaned_data"] if state["cleaned_data"] else state["raw_data"]
    sample_row = data_source[0]
    columns = list(sample_row.keys())
    
    # Remove target from columns list for LLM context
    columns_without_target = [col for col in columns if col != state["target_column"]]
    
    prompt = f"""
Create feature engineering plan:

Columns (excluding target): {columns_without_target}
Target: {state["target_column"]}

CRITICAL RULES:
1. NEVER include the target column '{state["target_column"]}' in any transformation
2. The target '{state["target_column"]}' must be dropped before feature engineering
3. DO NOT encode, scale, or transform the target column

Identify:
- Categorical columns for one-hot encoding
- Numeric columns for scaling
DO NOT modify target: {state["target_column"]}

JSON format:
{{
  "drop_columns": ["{state["target_column"]}"],
  "one_hot_columns": ["cat1", "cat2"],
  "scale_numeric": ["age", "salary"],
  "polynomial_features": [],
  "reasoning": "explanation"
}}
"""
    
    fe_plan = ask_ollama(prompt)
    
    if not fe_plan:
        state["errors"].append("Failed to get FE plan")
        fe_plan = {
            "drop_columns": [state["target_column"]],
            "one_hot_columns": [],
            "scale_numeric": [],
            "polynomial_features": [],
            "reasoning": "Default fallback"
        }
    
    # CRITICAL SAFETY: Ensure target is always in drop_columns
    if "drop_columns" not in fe_plan:
        fe_plan["drop_columns"] = []
    if state["target_column"] not in fe_plan["drop_columns"]:
        fe_plan["drop_columns"].append(state["target_column"])
    
    # Safety: don't encode/scale target
    if state["target_column"] in fe_plan.get("one_hot_columns", []):
        fe_plan["one_hot_columns"].remove(state["target_column"])
    if state["target_column"] in fe_plan.get("scale_numeric", []):
        fe_plan["scale_numeric"].remove(state["target_column"])
    if state["target_column"] in fe_plan.get("polynomial_features", []):
        fe_plan["polynomial_features"].remove(state["target_column"])
    
    state["fe_plan"] = fe_plan
    return state

async def fit_fe_node(state: MLPipelineState) -> MLPipelineState:
    """Node 8: Fit feature engineering on training data"""
    state["current_step"] = "Feature engineering (train)"
    
    # CRITICAL: Ensure target column is in the FE plan's drop list
    target = state["target_column"]
    if target not in state["fe_plan"].get("drop_columns", []):
        if "drop_columns" not in state["fe_plan"]:
            state["fe_plan"]["drop_columns"] = []
        state["fe_plan"]["drop_columns"].append(target)
    
    async with Client(mcp) as client:
        result = await client.call_tool("fit_feature_engineering", {
            "dataframe_json": state["train_data"],
            "fe_plan": state["fe_plan"]
        })
        output = await get_tool_output(result)
        
        state["train_fe_data"] = output["fe_data"]
        state["transform_params"] = output["transform_params"]
        state["column_names"] = output["column_names"]
        
        # SAFETY CHECK: Verify target is not in features
        if target in state["column_names"]:
            state["errors"].append(f"CRITICAL: Target column '{target}' found in features!")
            st.error(f"⚠️ Data leakage detected: '{target}' in features!")
    
    return state

async def transform_fe_node(state: MLPipelineState) -> MLPipelineState:
    """Node 9: Transform test data using fitted parameters"""
    state["current_step"] = "Feature engineering (test)"
    
    async with Client(mcp) as client:
        result = await client.call_tool("transform_feature_engineering", {
            "dataframe_json": state["test_data"],
            "fe_plan": state["fe_plan"],
            "transform_params": state["transform_params"]
        })
        output = await get_tool_output(result)
        
        state["test_fe_data"] = output["fe_data"]
    
    return state

async def llm_choose_model_node(state: MLPipelineState) -> MLPipelineState:
    """Node 10: Ollama chooses model"""
    state["current_step"] = "Choosing model"
    
    # Include previous attempt info if retrying
    retry_info = ""
    if state["retry_count"] > 0 and state.get("metrics"):
        retry_info = f"""
Previous attempt {state["retry_count"]}:
- Model: {state["model_choice"]["model_type"]}
- Accuracy: {state["metrics"].get("accuracy", 0):.2%}
- This was below threshold {state["accuracy_threshold"]:.2%}
CHOOSE A DIFFERENT MODEL!
"""
    
    prompt = f"""
Choose best model for classification:

Training samples: {state["num_rows_train"]}
Features: {len(state["train_fe_data"][0]) - 1}
Target: {state["target_column"]}

{retry_info}

Available:
- random_forest
- logistic_regression  
- decision_tree
- xgboost

JSON format:
{{
  "model_type": "random_forest",
  "params": {{"n_estimators": 100, "max_depth": 10, "random_state": 42}},
  "reasoning": "why?"
}}
"""
    
    model_choice = ask_ollama(prompt)
    
    if not model_choice:
        # Cycle through models on retry
        fallback_models = ["random_forest", "xgboost", "logistic_regression", "decision_tree"]
        model_idx = state["retry_count"] % len(fallback_models)
        model_choice = {
            "model_type": fallback_models[model_idx],
            "params": {"n_estimators": 50, "random_state": 42},
            "reasoning": "Default fallback"
        }
    
    state["model_choice"] = model_choice
    return state

async def train_model_node(state: MLPipelineState) -> MLPipelineState:
    """Node 11: Train model"""
    state["current_step"] = "Training model"
    
    async with Client(mcp) as client:
        result = await client.call_tool("train_model", {
            "model_type": state["model_choice"]["model_type"],
            "params": state["model_choice"]["params"],
            "train_data": state["train_fe_data"],
            "target_column": state["target_column"]
        })
        output = await get_tool_output(result)
        
        state["model_path"] = output["model_path"]
        state["task_type"] = output["task_type"]
        state["feature_importances"] = output.get("feature_importances")
    
    return state

async def evaluate_model_node(state: MLPipelineState) -> MLPipelineState:
    """Node 12: Evaluate model"""
    state["current_step"] = "Evaluating model"
    
    async with Client(mcp) as client:
        result = await client.call_tool("evaluate_model", {
            "model_path": state["model_path"],
            "test_data": state["test_fe_data"],
            "target_column": state["target_column"],
            "problem_type": state["task_type"]
        })
        state["metrics"] = await get_tool_output(result)
    
    return state

async def check_performance_node(state: MLPipelineState) -> MLPipelineState:
    """Node 13: Check if performance is acceptable"""
    state["current_step"] = "Checking performance"
    
    metrics = state["metrics"]
    
    # For classification, check accuracy
    if state["task_type"] == "classification":
        accuracy = metrics.get("accuracy", 0)
        
        if accuracy < state["accuracy_threshold"]:
            if state["retry_count"] < state["max_retries"]:
                state["needs_retry"] = True
                state["retry_count"] += 1
                state["warnings"].append(
                    f"Accuracy {accuracy:.2%} below threshold {state['accuracy_threshold']:.2%}. "
                    f"Retrying with different model (attempt {state['retry_count']}/{state['max_retries']})"
                )
            else:
                state["needs_retry"] = False
                state["warnings"].append(
                    f"Max retries reached. Final accuracy: {accuracy:.2%}"
                )
        else:
            state["needs_retry"] = False
    else:
        # For regression, always accept (can add R² threshold if needed)
        state["needs_retry"] = False
    
    return state

# ======================================
# CONDITIONAL ROUTING
# ======================================

def should_clean_data(state: MLPipelineState) -> Literal["clean", "skip_clean"]:
    """Decide whether to clean data based on assessment"""
    if state["skip_cleaning"]:
        return "skip_clean"
    return "clean"

def should_retry_model(state: MLPipelineState) -> Literal["retry", "finish"]:
    """Decide whether to retry with a different model"""
    if state["needs_retry"]:
        return "retry"
    return "finish"

# ======================================
# BUILD LANGGRAPH
# ======================================

def build_workflow() -> StateGraph:
    """Build the LangGraph workflow with conditional logic"""
    workflow = StateGraph(MLPipelineState)
    
    # Add nodes
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("extract_schema", extract_schema_node)
    workflow.add_node("assess_quality", assess_data_quality_node)
    workflow.add_node("llm_clean", llm_plan_cleaning_node)
    workflow.add_node("clean", clean_data_node)
    workflow.add_node("split", split_data_node)
    workflow.add_node("llm_fe", llm_plan_fe_node)
    workflow.add_node("fit_fe", fit_fe_node)
    workflow.add_node("transform_fe", transform_fe_node)
    workflow.add_node("llm_model", llm_choose_model_node)
    workflow.add_node("train", train_model_node)
    workflow.add_node("evaluate", evaluate_model_node)
    workflow.add_node("check_performance", check_performance_node)
    
    # Linear flow until quality assessment
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "extract_schema")
    workflow.add_edge("extract_schema", "assess_quality")
    
    # Conditional: Clean or skip cleaning
    workflow.add_conditional_edges(
        "assess_quality",
        should_clean_data,
        {
            "clean": "llm_clean",
            "skip_clean": "split"
        }
    )
    
    # Cleaning path
    workflow.add_edge("llm_clean", "clean")
    workflow.add_edge("clean", "split")
    
    # Feature engineering & training
    workflow.add_edge("split", "llm_fe")
    workflow.add_edge("llm_fe", "fit_fe")
    workflow.add_edge("fit_fe", "transform_fe")
    workflow.add_edge("transform_fe", "llm_model")
    workflow.add_edge("llm_model", "train")
    workflow.add_edge("train", "evaluate")
    workflow.add_edge("evaluate", "check_performance")
    
    # Conditional: Retry or finish
    workflow.add_conditional_edges(
        "check_performance",
        should_retry_model,
        {
            "retry": "llm_model",  # Loop back to try different model
            "finish": END
        }
    )
    
    return workflow.compile()

# ======================================
# STREAMLIT UI
# ======================================

async def run_ml_pipeline(filename: str, target_column: str, accuracy_threshold: float = 0.7):
    """Execute the LangGraph ML pipeline with conditional logic"""
    
    # Initialize state
    initial_state: MLPipelineState = {
        "filename": filename,
        "target_column": target_column,
        "raw_data": None,
        "schema": None,
        "cleaning_plan": None,
        "cleaned_data": None,
        "train_data": None,
        "test_data": None,
        "fe_plan": None,
        "train_fe_data": None,
        "test_fe_data": None,
        "transform_params": None,
        "model_choice": None,
        "model_path": None,
        "metrics": None,
        "task_type": None,
        "column_names": None,
        "num_cols": None,
        "num_rows_train": None,
        "num_rows_test": None,
        "feature_importances": None,
        "retry_count": 0,
        "max_retries": 3,
        "needs_retry": False,
        "skip_cleaning": False,
        "accuracy_threshold": accuracy_threshold,
        "errors": [],
        "warnings": [],
        "current_step": "Starting"
    }
    
    # Build graph
    app = build_workflow()
    
    # Track which steps have been executed
    executed_steps = {}
    
    final_state = None
    current_state = None
    step_counter = 0
    
    # Stream through graph
    async for event in app.astream(initial_state):
        node_name = list(event.keys())[0]
        
        if node_name == "__end__":
            final_state = event["__end__"]
            break
        
        state = event[node_name]
        current_state = state  # Keep track of the latest state
        
        # Create or update status for this step
        if node_name not in executed_steps:
            step_counter += 1
            status_labels = {
                "load_data": "📂 Loading data",
                "extract_schema": "📊 Analyzing schema",
                "assess_quality": "🔍 Assessing data quality",
                "llm_clean": "🧠 Ollama planning data cleaning",
                "clean": "🧹 Cleaning data",
                "split": "✂️ Splitting data",
                "llm_fe": "🧠 Ollama planning feature engineering",
                "fit_fe": "🔧 Feature engineering (train)",
                "transform_fe": "🔧 Feature engineering (test)",
                "llm_model": "🧠 Ollama choosing model",
                "train": "🎯 Training model",
                "evaluate": "📈 Evaluating model",
                "check_performance": "✅ Checking performance"
            }
            
            label = status_labels.get(node_name, node_name)
            if state.get("retry_count", 0) > 0 and node_name == "llm_model":
                label = f"🔄 Retry {state['retry_count']}: Choosing different model"
            
            executed_steps[node_name] = st.status(f"{label}...", expanded=True)
        
        status = executed_steps[node_name]
        
        with status:
            if node_name == "load_data" and state.get("raw_data"):
                st.write(f"✓ Loaded {state['num_rows_train']} rows, {state['num_cols']} columns")
            
            elif node_name == "extract_schema" and state.get("schema"):
                st.write(f"✓ Found {state['schema']['num_cols']} columns")
                with st.expander("View Schema"):
                    for col in state["schema"]["columns"]:
                        st.write(f"**{col['name']}** - {col['suggested_type']} "
                                f"({col['missing_pct']*100:.1f}% missing)")
            
            elif node_name == "assess_quality":
                if state.get("skip_cleaning"):
                    st.write("✓ Data quality is good - skipping cleaning")
                else:
                    st.write("✓ Data needs cleaning")
            
            elif node_name == "llm_clean" and state.get("cleaning_plan"):
                st.write("✓ Cleaning plan created")
                with st.expander("View Cleaning Plan"):
                    st.json(state["cleaning_plan"])
            
            elif node_name == "clean" and state.get("cleaned_data"):
                st.write(f"✓ Cleaned: {state['num_rows_train']} rows, {state['num_cols']} columns")
            
            elif node_name == "split" and state.get("train_data"):
                st.write(f"✓ Train: {state['num_rows_train']} rows")
                st.write(f"✓ Test: {state['num_rows_test']} rows")
            
            elif node_name == "llm_fe" and state.get("fe_plan"):
                st.write("✓ FE plan created")
                with st.expander("View FE Plan"):
                    st.json(state["fe_plan"])
            
            elif node_name == "fit_fe" and state.get("train_fe_data"):
                st.write(f"✓ Created {len(state['column_names'])} features")
                with st.expander("View Features"):
                    st.write(state["column_names"])
            
            elif node_name == "transform_fe" and state.get("test_fe_data"):
                st.write(f"✓ Test data transformed: {len(state['test_fe_data'][0])} features")
            
            elif node_name == "llm_model" and state.get("model_choice"):
                retry_text = f" (Retry {state['retry_count']})" if state['retry_count'] > 0 else ""
                st.write(f"✓ Model chosen{retry_text}: {state['model_choice']['model_type']}")
                with st.expander("View Model Choice"):
                    st.json(state["model_choice"])
            
            elif node_name == "train" and state.get("model_path"):
                st.write(f"✓ Model trained and saved")
            
            elif node_name == "evaluate" and state.get("metrics"):
                st.write("✓ Evaluation complete!")
                if state["task_type"] == "classification":
                    st.write(f"Accuracy: {state['metrics']['accuracy']:.2%}")
            
            elif node_name == "check_performance":
                if state.get("needs_retry"):
                    st.warning(f"⚠️ Performance below threshold - retrying with different model")
                else:
                    st.write("✓ Performance acceptable!")
        
        # Update status with completion message
        node_display = node_name.replace('_', ' ').title()
        if state.get("needs_retry") and node_name == "check_performance":
            status.update(label="🔄 Retrying with different model", state="running")
        else:
            status.update(label=f"✅ {node_display} complete!", state="complete")
    
    # Use the last state if __end__ wasn't caught
    if final_state is None and current_state is not None:
        final_state = current_state
    
    # Show warnings if any
    if final_state and final_state.get("warnings"):
        st.warning("⚠️ Warnings:")
        for warning in final_state["warnings"]:
            st.write(f"- {warning}")
    
    # Show final results
    if final_state and final_state.get("metrics"):
        st.markdown("---")
        st.success("🎉 Pipeline completed successfully!")
        
        st.subheader("📊 Final Model Performance")
        task_type = final_state["task_type"]
        metrics = final_state["metrics"]
        
        if task_type == "classification":
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['precision']:.2%}")
            col3.metric("Recall", f"{metrics['recall']:.2%}")
            col4.metric("F1 Score", f"{metrics['f1']:.2%}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{metrics['mse']:.2f}")
            col2.metric("MAE", f"{metrics['mae']:.2f}")
            col3.metric("R²", f"{metrics['r2']:.2f}")
        
        # Show retry history
        if final_state["retry_count"] > 0:
            st.info(f"ℹ️ Model was retried {final_state['retry_count']} time(s) to achieve better performance")
        
        # Show model info
        st.subheader("🎯 Model Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Model Type:** {final_state['model_choice']['model_type']}")
            st.write(f"**Task:** {final_state['task_type']}")
            st.write(f"**Features:** {len(final_state['column_names'])}")
        with col2:
            st.write(f"**Training Samples:** {final_state['num_rows_train']}")
            st.write(f"**Test Samples:** {final_state['num_rows_test']}")
            st.write(f"**Model Path:** `{final_state['model_path']}`")
        
        # Auto-save results to file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        result_data = {
            "timestamp": timestamp,
            "filename": final_state["filename"],
            "target": final_state["target_column"],
            "train_size": final_state["num_rows_train"],
            "test_size": final_state["num_rows_test"],
            "num_features": len(final_state["column_names"]),
            "features": final_state["column_names"],
            "model": final_state["model_choice"],
            "model_path": final_state["model_path"],
            "task_type": final_state["task_type"],
            "metrics": final_state["metrics"],
            "cleaning_plan": final_state.get("cleaning_plan"),
            "fe_plan": final_state["fe_plan"],
            "retry_count": final_state["retry_count"],
            "warnings": final_state["warnings"],
            "feature_importances": final_state.get("feature_importances")
        }
        
        results_filename = f"pipeline_results_{timestamp}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        with open(results_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        st.success(f"✅ Results automatically saved to: `{results_path}`")
        
        # Show feature importances if available
        if final_state.get("feature_importances"):
            with st.expander("📈 Feature Importances"):
                import pandas as pd
                fi_df = pd.DataFrame(
                    list(final_state["feature_importances"].items()),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                st.dataframe(fi_df, use_container_width=True)
        
        # Download button
        with st.expander("💾 Download Results"):
            st.download_button(
                "Download Results (JSON)",
                data=json.dumps(result_data, indent=2),
                file_name=results_filename,
                mime="application/json"
            )

def main():
    st.set_page_config(
        page_title="🤖 AI ML Pipeline (LangGraph)",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered ML Pipeline (LangGraph)")
    st.markdown("""
    **LangGraph orchestrates with intelligent decisions**  
    **Conditional Logic:** Skips cleaning if data is good  
    **Auto-Retry:** Tries different models if performance is poor  
    **Ollama** analyzes your data and makes all decisions  
    **MCP Tools** execute the work  
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🧠 Ollama Status")
        try:
            ollama.chat(
                model='qwen2.5-coder:7b',
                messages=[{'role': 'user', 'content': 'test'}]
            )
            st.success("✓ Ollama connected (qwen2.5-coder:7b)")
        except Exception as e:
            st.error("❌ Ollama not running!")
            st.code("ollama serve")
            st.stop()
        
        st.markdown("---")
        st.subheader("🎯 Performance Settings")
        accuracy_threshold = st.slider(
            "Accuracy Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="If accuracy is below this, try a different model"
        )
        
        st.markdown("---")
        st.subheader("📖 How It Works")
        st.markdown("""
        **Conditional Flow:**
        1. Load Data
        2. Extract Schema
        3. **Assess Quality** → Skip or Clean
        4. Train/Test Split
        5. Plan Features
        6. Feature Engineering
        7. **Choose Model**
        8. Train & Evaluate
        9. **Check Performance** → Retry if poor
        
        **Max 3 retries with different models**
        """)
    
    # Main content
    st.header("1️⃣ Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file:
        # Save file
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✓ File saved: {uploaded_file.name}")
        
        # Preview
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10))
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Select target
        st.header("2️⃣ Select Target Column")
        target_col = st.selectbox(
            "Which column do you want to predict?",
            options=df.columns,
            help="This is the column your model will try to predict"
        )
        
        st.info(f"🎯 Target: **{target_col}**")
        
        # Run pipeline
        st.header("3️⃣ Run Pipeline")
        
        if st.button("🚀 Start AI Pipeline (LangGraph)", type="primary", use_container_width=True):
            st.markdown("---")
            
            try:
                # Run async pipeline
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    run_ml_pipeline(uploaded_file.name, target_col, accuracy_threshold)
                )
                loop.close()
                
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                with st.expander("🐛 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
