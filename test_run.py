# test_run.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from core.orchestrator import Orchestrator
from core.memory_system import MemorySystem
from utils.data_validation import DataValidator
from agents.data_engineer import DataEngineer
from agents.viz_specialist import VizSpecialist
from core.function_registry import FunctionRegistry


def create_sample_data():
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 100, 100),
        'customers': np.random.randint(50, 200, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    return data


def print_result(result, truncate_keys=["image_data"]):
    """Helper function to print results with truncated fields"""
    if isinstance(result, dict):
        # Create a copy of the result to avoid modifying the original
        result_copy = result.copy()

        # Truncate specific keys
        for key in truncate_keys:
            if key in result_copy and isinstance(result_copy[key], str) and len(result_copy[key]) > 100:
                result_copy[key] = result_copy[key][:30] + "..." + result_copy[key][-30:]

        # Check for nested dictionaries
        for key, value in result_copy.items():
            if isinstance(value, dict):
                result_copy[key] = print_result(value, truncate_keys)
            elif isinstance(value, list):
                result_copy[key] = [print_result(item, truncate_keys) if isinstance(item, dict) else item for item in
                                    value]

        return result_copy
    else:
        return result


def main():
    # Initialize components
    memory_system = MemorySystem()
    function_registry = FunctionRegistry()

    # Register necessary functions with the function registry
    register_data_functions(function_registry)
    register_viz_functions(function_registry)

    # Initialize orchestrator
    orchestrator = Orchestrator(
        memory_system=memory_system,
        function_registry=function_registry
    )

    # Create sample data
    data = create_sample_data()

    print("\n=== Testing Data Processing ===")
    # Test data processing with validation rules
    data_request = {
        "type": "data_processing",
        "operation": "load",
        "params": {
            "data": data,
            "validation_rules": {
                "sales": {
                    "numeric": {},
                    "range": {"min": 0, "max": 2000},
                    "missing": {"threshold": 0.1}
                },
                "customers": {
                    "numeric": {},
                    "range": {"min": 0},
                    "missing": {"threshold": 0.1}
                },
                "category": {
                    "categorical": {"categories": ['A', 'B', 'C']},
                    "missing": {"threshold": 0}
                },
                "date": {
                    "datetime": {},
                    "missing": {"threshold": 0}
                }
            }
        }
    }
    result = orchestrator.process_request(data_request)
    print("Data loading result:", print_result(result))

    # Test data transformation
    transform_request = {
        "type": "data_processing",
        "operation": "transform",
        "params": {
            "operations": [
                {"type": "filter", "column": "sales", "condition": ">", "value": 500},
                {"type": "aggregate", "group_by": "category", "metrics": ["mean", "sum"]}
            ],
            "validation_rules": {
                "sales": {
                    "numeric": {},
                    "range": {"min": 500}
                }
            }
        }
    }
    result = orchestrator.process_request(transform_request)
    print("\nTransformation result:", print_result(result))

    # Store the data for visualizations
    # This is important to ensure data is available in memory
    store_data_request = {
        "type": "memory",
        "operation": "store",
        "params": {
            "key": "test_data",
            "data": data
        }
    }
    orchestrator.process_request(store_data_request)

    print("\n=== Testing Visualizations ===")
    viz_types = ["line", "scatter", "bar", "histogram"]
    for viz_type in viz_types:
        # Explicitly store data before each visualization test to ensure it's available
        orchestrator.process_request(store_data_request)

        # Create the visualization
        viz_request = {
            "type": "visualization",
            "viz_type": viz_type,
            "params": {
                "x": "date" if viz_type in ["line", "bar"] else "sales",
                "y": "sales" if viz_type in ["line", "bar"] else "customers",
                "title": f"Test {viz_type} plot",
                "color": "region" if viz_type != "scatter" else None
            }
        }
        result = orchestrator.process_request(viz_request)
        print(f"\n{viz_type} visualization result:", print_result(result))

    print("\n=== Testing Workflow ===")
    # Make sure the data is stored in memory before workflow test
    orchestrator.process_request(store_data_request)

    workflow_request = {
        "type": "workflow",
        "workflow_type": "basic",
        "steps": [
            {
                "type": "data_operation",
                "operation": "filter",
                "params": {"column": "sales", "condition": ">", "value": 900}
            },
            {
                "type": "visualization",
                "viz_type": "scatter",
                "params": {"x": "sales", "y": "customers"}
            }
        ]
    }
    result = orchestrator.process_request(workflow_request)
    print("Workflow execution result:", print_result(result))

    print("\n=== Testing Memory Operations ===")
    # Test storing data
    store_request = {
        "type": "memory",
        "operation": "store",
        "params": {
            "key": "test_data",
            "data": data
        }
    }
    result = orchestrator.process_request(store_request)
    print("Memory store result:", print_result(result))

    # Test retrieving data
    retrieve_request = {
        "type": "memory",
        "operation": "retrieve",
        "params": {
            "key": "test_data"
        }
    }
    result = orchestrator.process_request(retrieve_request)
    print("\nMemory retrieve result:", print_result(result))

    print("\n=== Testing Agent Management ===")
    # Test agent status
    status_request = {
        "type": "agent_management",
        "operation": "status"
    }
    result = orchestrator.process_request(status_request)
    print("Agent status:", print_result(result))

    # Test system configuration
    config_request = {
        "type": "system_config",
        "config_type": "system",
        "updates": {
            "memory": {
                "max_cache_size": 1000
            }
        }
    }
    result = orchestrator.process_request(config_request)
    print("\nSystem configuration update result:", print_result(result))

    # Test conversation summary
    summary = orchestrator.get_conversation_summary()
    print("\nConversation summary:", print_result(summary))

    # Test state saving
    save_state_result = orchestrator.save_state("data/orchestrator_state.json")
    print("\nState save result:", print_result(save_state_result))


def register_data_functions(registry):
    """Register data-related functions with the function registry"""

    def data_filter(**params):
        """Filter DataFrame based on conditions"""
        try:
            df = params.get("dataframe")
            column = params.get("column")
            condition = params.get("condition")
            value = params.get("value")

            if df is None or column is None or condition is None:
                return {"success": False, "error": "Missing required parameters"}

            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in DataFrame"}

            condition_map = {
                ">": lambda x, v: x > v,
                ">=": lambda x, v: x >= v,
                "<": lambda x, v: x < v,
                "<=": lambda x, v: x <= v,
                "==": lambda x, v: x == v,
                "!=": lambda x, v: x != v,
                "in": lambda x, v: x.isin(v),
                "not in": lambda x, v: ~x.isin(v),
                "contains": lambda x, v: x.str.contains(str(v), na=False),
                "not contains": lambda x, v: ~x.str.contains(str(v), na=False)
            }

            if condition not in condition_map:
                return {"success": False, "error": f"Unsupported condition: {condition}"}

            result_df = df[condition_map[condition](df[column], value)]

            return {
                "success": True,
                "dataframe": result_df,
                "filtered_rows": len(result_df)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def data_aggregate(**params):
        """Aggregate DataFrame"""
        try:
            df = params.get("dataframe")
            group_by = params.get("group_by")
            metrics = params.get("metrics", ["mean"])

            if df is None or group_by is None:
                return {"success": False, "error": "Missing required parameters"}

            if isinstance(group_by, str):
                group_by = [group_by]

            if not all(col in df.columns for col in group_by):
                return {"success": False, "error": f"One or more grouping columns not found in dataset"}

            # Simple aggregation
            result_df = df.groupby(group_by).agg(metrics).reset_index()

            return {
                "success": True,
                "dataframe": result_df,
                "group_count": len(result_df)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def data_sort(**params):
        """Sort DataFrame"""
        try:
            df = params.get("dataframe")
            columns = params.get("columns")
            ascending = params.get("ascending", True)

            if df is None or columns is None:
                return {"success": False, "error": "Missing required parameters"}

            if isinstance(columns, str):
                columns = [columns]

            if not all(col in df.columns for col in columns):
                return {"success": False, "error": f"One or more sort columns not found in dataset"}

            result_df = df.sort_values(by=columns, ascending=ascending)

            return {
                "success": True,
                "dataframe": result_df
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def data_select(**params):
        """Select columns from DataFrame"""
        try:
            df = params.get("dataframe")
            columns = params.get("columns", [])

            if df is None:
                return {"success": False, "error": "Missing required parameters"}

            if not columns:
                return {"success": True, "dataframe": df}

            if not all(col in df.columns for col in columns):
                return {"success": False, "error": f"One or more columns not found in dataset"}

            result_df = df[columns]

            return {
                "success": True,
                "dataframe": result_df
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # Register all data functions
    registry.register_function(data_filter, "data", "filter")
    registry.register_function(data_filter, "data", "data_filter")
    registry.register_function(data_aggregate, "data", "aggregate")
    registry.register_function(data_aggregate, "data", "data_aggregate")
    registry.register_function(data_sort, "data", "sort")
    registry.register_function(data_sort, "data", "data_sort")
    registry.register_function(data_select, "data", "select")
    registry.register_function(data_select, "data", "data_select")


def register_viz_functions(registry):
    """Register visualization functions with the function registry"""

    def viz_line(**params):
        """Create a line plot visualization"""
        try:
            data = params.get("data")
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Line Plot")

            if data is None or x is None or y is None:
                return {"success": False, "error": "Missing required parameters"}

            if x not in data.columns or y not in data.columns:
                return {"success": False,
                        "error": f"Columns not found: {x if x not in data.columns else ''} {y if y not in data.columns else ''}".strip()}

            # In a real implementation, this would create the plot
            return {
                "success": True,
                "message": "Line plot created successfully",
                "plot_data": {
                    "type": "line",
                    "x": x,
                    "y": y,
                    "title": title,
                    "data_shape": data.shape
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def viz_scatter(**params):
        """Create a scatter plot visualization"""
        try:
            data = params.get("data")
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Scatter Plot")

            if data is None or x is None or y is None:
                return {"success": False, "error": "Missing required parameters"}

            if x not in data.columns or y not in data.columns:
                return {"success": False,
                        "error": f"Columns not found: {x if x not in data.columns else ''} {y if y not in data.columns else ''}".strip()}

            # In a real implementation, this would create the plot
            return {
                "success": True,
                "message": "Scatter plot created successfully",
                "plot_data": {
                    "type": "scatter",
                    "x": x,
                    "y": y,
                    "title": title,
                    "data_shape": data.shape
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def viz_bar(**params):
        """Create a bar plot visualization"""
        try:
            data = params.get("data")
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Bar Plot")

            if data is None or x is None or y is None:
                return {"success": False, "error": "Missing required parameters"}

            if x not in data.columns or y not in data.columns:
                return {"success": False,
                        "error": f"Columns not found: {x if x not in data.columns else ''} {y if y not in data.columns else ''}".strip()}

            # In a real implementation, this would create the plot
            return {
                "success": True,
                "message": "Bar plot created successfully",
                "plot_data": {
                    "type": "bar",
                    "x": x,
                    "y": y,
                    "title": title,
                    "data_shape": data.shape
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def viz_histogram(**params):
        """Create a histogram visualization"""
        try:
            data = params.get("data")
            x = params.get("x")
            bins = params.get("bins", 10)
            title = params.get("title", "Histogram")

            if data is None or x is None:
                return {"success": False, "error": "Missing required parameters"}

            if x not in data.columns:
                return {"success": False, "error": f"Column not found: {x}"}

            # In a real implementation, this would create the plot
            return {
                "success": True,
                "message": "Histogram created successfully",
                "plot_data": {
                    "type": "histogram",
                    "x": x,
                    "bins": bins,
                    "title": title,
                    "data_shape": data.shape
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Register all visualization functions
    registry.register_function(viz_line, "visualization", "viz_line")
    registry.register_function(viz_scatter, "visualization", "viz_scatter")
    registry.register_function(viz_bar, "visualization", "viz_bar")
    registry.register_function(viz_histogram, "visualization", "viz_histogram")


if __name__ == "__main__":
    main()