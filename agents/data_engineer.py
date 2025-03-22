# agents/data_engineer.py

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging


class DataEngineer:
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.current_dataset = None
        self.operation_history = []
        self.status = "active"
        self.logger = self._setup_logger()
        self.supported_operations = {
            "filter": self._apply_filter,
            "aggregate": self._apply_aggregation,
            "sort": self._apply_sort,
            "transform": self._apply_transform,
            "join": self._apply_join,
            "select": self._apply_select,
            "rename": self._apply_rename,
            "dropna": self._apply_dropna,
            "fillna": self._apply_fillna
        }
        self.supported_transformations = {
            "log": np.log,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "round": np.round,
            "standardize": self._standardize,
            "normalize": self._normalize,
            "binary": self._to_binary,
            "categorical": self._to_categorical
        }

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        Path('output/logs').mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler('output/logs/data_engineer.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_data(self, data: Union[pd.DataFrame, Dict, List, str]) -> Dict[str, Any]:
        """
        Load data into the system from various sources

        Parameters:
        - data: Can be DataFrame, Dict, List, or file path string

        Returns:
        - Dictionary with operation result
        """
        try:
            if isinstance(data, pd.DataFrame):
                self.current_dataset = data.copy()
            elif isinstance(data, str):
                # Handle file path
                file_path = Path(data)
                if not file_path.exists():
                    return {"success": False, "error": f"File not found: {data}"}

                if file_path.suffix == '.csv':
                    self.current_dataset = pd.read_csv(file_path)
                elif file_path.suffix in ['.xlsx', '.xls']:
                    self.current_dataset = pd.read_excel(file_path)
                elif file_path.suffix == '.json':
                    self.current_dataset = pd.read_json(file_path)
                else:
                    return {"success": False, "error": f"Unsupported file format: {file_path.suffix}"}
            else:
                try:
                    self.current_dataset = pd.DataFrame(data)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Unable to convert data to DataFrame: {str(e)}"
                    }

            # Record operation in memory system
            operation_details = {
                "shape": self.current_dataset.shape,
                "columns": list(self.current_dataset.columns),
                "dtypes": self.current_dataset.dtypes.astype(str).to_dict(),
                "timestamp": datetime.now().isoformat()
            }

            self.memory_system.add_operation(
                operation_type="data_load",
                details=operation_details,
                result={"success": True}
            )

            return {
                "success": True,
                "message": "Data loaded successfully",
                "shape": self.current_dataset.shape,
                "columns": list(self.current_dataset.columns),
                "dtypes": self.current_dataset.dtypes.astype(str).to_dict()
            }
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_data(self, file_path: str, format: str = None) -> Dict[str, Any]:
        """
        Save current dataset to file

        Parameters:
        - file_path: Path to save the file
        - format: Optional format override

        Returns:
        - Dictionary with operation result
        """
        try:
            if self.current_dataset is None:
                return {"success": False, "error": "No dataset to save"}

            path = Path(file_path)
            format = format or path.suffix.lstrip('.')

            if format == 'csv':
                self.current_dataset.to_csv(path, index=False)
            elif format in ['xlsx', 'xls']:
                self.current_dataset.to_excel(path, index=False)
            elif format == 'json':
                self.current_dataset.to_json(path)
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}

            self.memory_system.add_operation(
                operation_type="data_save",
                details={"file_path": str(path), "format": format},
                result={"success": True}
            )

            return {
                "success": True,
                "message": f"Data saved successfully to {path}",
                "format": format
            }
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return {"success": False, "error": str(e)}

    def transform_data(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply transformations to the current dataset

        Parameters:
        - operations: List of operation dictionaries

        Returns:
        - Dictionary with operation result
        """
        try:
            if self.current_dataset is None:
                return {"success": False, "error": "No dataset loaded"}

            result_df = self.current_dataset.copy()
            applied_operations = []
            operation_results = []

            for operation in operations:
                op_type = operation.get("type")
                if op_type not in self.supported_operations:
                    return {
                        "success": False,
                        "error": f"Unsupported operation: {op_type}"
                    }

                try:
                    result_df = self.supported_operations[op_type](result_df, operation)
                    applied_operations.append(operation)
                    operation_results.append({
                        "operation": op_type,
                        "success": True,
                        "resulting_shape": result_df.shape
                    })
                except Exception as e:
                    operation_results.append({
                        "operation": op_type,
                        "success": False,
                        "error": str(e)
                    })
                    return {
                        "success": False,
                        "error": f"Error in {op_type} operation: {str(e)}",
                        "operation_results": operation_results
                    }

            self.current_dataset = result_df

            # Record transformation in memory system
            self.memory_system.add_operation(
                operation_type="data_transform",
                details={
                    "operations": applied_operations,
                    "initial_shape": self.current_dataset.shape,
                    "final_shape": result_df.shape
                },
                result={"success": True, "operation_results": operation_results}
            )

            return {
                "success": True,
                "message": "Transformations applied successfully",
                "initial_shape": self.current_dataset.shape,
                "final_shape": result_df.shape,
                "operations_applied": applied_operations,
                "operation_results": operation_results
            }
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            return {"success": False, "error": str(e)}

    def _apply_filter(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filter operation

        Parameters:
        - df: Input DataFrame
        - operation: Filter operation parameters

        Returns:
        - Filtered DataFrame
        """
        column = operation.get("column")
        condition = operation.get("condition")
        value = operation.get("value")

        if not column in df.columns:
            raise ValueError(f"Column {column} not found in dataset")

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
            "not contains": lambda x, v: ~x.str.contains(str(v), na=False),
            "between": lambda x, v: x.between(v[0], v[1]),
            "isna": lambda x, v: x.isna(),
            "notna": lambda x, v: x.notna(),
            "startswith": lambda x, v: x.str.startswith(str(v)),
            "endswith": lambda x, v: x.str.endswith(str(v))
        }

        if condition not in condition_map:
            raise ValueError(f"Unsupported condition: {condition}")

        return df[condition_map[condition](df[column], value)]

    def _apply_aggregation(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply aggregation operation

        Parameters:
        - df: Input DataFrame
        - operation: Aggregation operation parameters

        Returns:
        - Aggregated DataFrame
        """
        group_by = operation.get("group_by")
        metrics = operation.get("metrics", ["mean"])

        if isinstance(group_by, str):
            group_by = [group_by]

        if not all(col in df.columns for col in group_by):
            raise ValueError(f"One or more grouping columns not found in dataset")

        agg_map = {
            "mean": "mean",
            "sum": "sum",
            "count": "count",
            "min": "min",
            "max": "max",
            "std": "std",
            "var": "var",
            "first": "first",
            "last": "last",
            "median": "median",
            "nunique": "nunique",
            "size": "size"
        }

        # Create a grouped DataFrame
        grouped = df.groupby(group_by)

        # Handle metrics based on type
        if isinstance(metrics, list) and all(isinstance(m, str) for m in metrics):
            # List of string metrics - apply to all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {}
            for col in numeric_cols:
                agg_dict[col] = [agg_map[m] for m in metrics if m in agg_map]

            if not agg_dict:
                raise ValueError("No numeric columns available for aggregation")

            result = grouped.agg(agg_dict)
            # Flatten multi-level columns if they exist
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = [f"{col}_{agg}" for col, agg in result.columns]

        elif isinstance(metrics, dict):
            # Dictionary mapping columns to aggregation functions
            agg_dict = {}
            for col, aggs in metrics.items():
                if col in df.columns:
                    if isinstance(aggs, list):
                        agg_dict[col] = [agg_map[a] for a in aggs if a in agg_map]
                    else:
                        agg_dict[col] = agg_map[aggs] if aggs in agg_map else aggs

            if not agg_dict:
                raise ValueError("No valid column-aggregation pairs provided")

            result = grouped.agg(agg_dict)
            # Flatten multi-level columns if they exist
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = [f"{col}_{agg}" for col, agg in result.columns]

        else:
            raise ValueError("Metrics must be a list of strings or a dictionary mapping columns to aggregations")

        # Reset index to convert group by columns back to regular columns
        return result.reset_index()

    def _apply_sort(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply sort operation

        Parameters:
        - df: Input DataFrame
        - operation: Sort operation parameters

        Returns:
        - Sorted DataFrame
        """
        columns = operation.get("columns")
        ascending = operation.get("ascending", True)

        if isinstance(columns, str):
            columns = [columns]
            if isinstance(ascending, bool):
                ascending = [ascending]

        if not all(col in df.columns for col in columns):
            raise ValueError(f"One or more sort columns not found in dataset")

        return df.sort_values(by=columns, ascending=ascending)

    def _apply_transform(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply transformation operation

        Parameters:
        - df: Input DataFrame
        - operation: Transform operation parameters

        Returns:
        - Transformed DataFrame
        """
        column = operation.get("column")
        transform_type = operation.get("transform_type")
        new_column = operation.get("new_column", column)
        params = operation.get("params", {})

        if not column in df.columns:
            raise ValueError(f"Column {column} not found in dataset")

        if transform_type not in self.supported_transformations:
            raise ValueError(f"Unsupported transformation: {transform_type}")

        result_df = df.copy()
        try:
            if transform_type in ["standardize", "normalize"]:
                result_df[new_column] = self.supported_transformations[transform_type](df[column], **params)
            else:
                result_df[new_column] = self.supported_transformations[transform_type](df[column])
        except Exception as e:
            raise ValueError(f"Error applying transformation: {str(e)}")

        return result_df

    def _apply_join(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply join operation

        Parameters:
        - df: Input DataFrame
        - operation: Join operation parameters

        Returns:
        - Joined DataFrame
        """
        right_df = operation.get("right_df")
        how = operation.get("how", "inner")
        on = operation.get("on")
        left_on = operation.get("left_on")
        right_on = operation.get("right_on")

        if not isinstance(right_df, pd.DataFrame):
            raise ValueError("right_df must be a DataFrame")

        if how not in ["inner", "left", "right", "outer"]:
            raise ValueError(f"Unsupported join type: {how}")

        if on and (left_on or right_on):
            raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on'")

        if on:
            return df.merge(right_df, on=on, how=how)
        elif left_on and right_on:
            return df.merge(right_df, left_on=left_on, right_on=right_on, how=how)
        else:
            raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

    def _apply_select(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply column selection operation

        Parameters:
        - df: Input DataFrame
        - operation: Select operation parameters

        Returns:
        - DataFrame with selected columns
        """
        columns = operation.get("columns", [])
        exclude = operation.get("exclude", [])

        if not columns and not exclude:
            raise ValueError("Must specify either columns to include or exclude")

        if columns:
            if not all(col in df.columns for col in columns):
                raise ValueError("One or more selected columns not found in dataset")
            return df[columns]
        else:
            if not all(col in df.columns for col in exclude):
                raise ValueError("One or more excluded columns not found in dataset")
            return df.drop(columns=exclude)

    def _apply_rename(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply column rename operation

        Parameters:
        - df: Input DataFrame
        - operation: Rename operation parameters

        Returns:
        - DataFrame with renamed columns
        """
        column_map = operation.get("columns", {})

        if not all(col in df.columns for col in column_map.keys()):
            raise ValueError("One or more columns to rename not found in dataset")

        return df.rename(columns=column_map)

    def _apply_dropna(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply drop NA operation

        Parameters:
        - df: Input DataFrame
        - operation: Drop NA operation parameters

        Returns:
        - DataFrame with NA values dropped
        """
        subset = operation.get("subset", None)
        how = operation.get("how", "any")

        if subset and not all(col in df.columns for col in subset):
            raise ValueError("One or more columns in subset not found in dataset")

        return df.dropna(subset=subset, how=how)

    def _apply_fillna(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply fill NA operation

        Parameters:
        - df: Input DataFrame
        - operation: Fill NA operation parameters

        Returns:
        - DataFrame with NA values filled
        """
        value = operation.get("value")
        method = operation.get("method")
        columns = operation.get("columns")

        result_df = df.copy()

        if columns:
            if not all(col in df.columns for col in columns):
                raise ValueError("One or more columns not found in dataset")
            for col in columns:
                if method:
                    result_df[col] = result_df[col].fillna(method=method)
                else:
                    result_df[col] = result_df[col].fillna(value)
        else:
            if method:
                result_df = result_df.fillna(method=method)
            else:
                result_df = result_df.fillna(value)

        return result_df

    def _standardize(self, series: pd.Series, **kwargs) -> pd.Series:
        """
        Standardize a series (z-score normalization)

        Parameters:
        - series: Input Series
        - kwargs: Additional parameters

        Returns:
        - Standardized Series
        """
        return (series - series.mean()) / series.std()

    def _normalize(self, series: pd.Series, **kwargs) -> pd.Series:
        """
        Normalize a series (min-max normalization)

        Parameters:
        - series: Input Series
        - kwargs: Additional parameters

        Returns:
        - Normalized Series
        """
        return (series - series.min()) / (series.max() - series.min())

    def _to_binary(self, series: pd.Series) -> pd.Series:
        """
        Convert series to binary values

        Parameters:
        - series: Input Series

        Returns:
        - Binary Series
        """
        return pd.get_dummies(series, prefix=series.name, drop_first=True)

    def _to_categorical(self, series: pd.Series) -> pd.Series:
        """
        Convert series to categorical type

        Parameters:
        - series: Input Series

        Returns:
        - Categorical Series
        """
        return pd.Categorical(series)

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the current dataset

        Returns:
        - Dictionary with dataset information
        """
        if self.current_dataset is None:
            return {"success": False, "error": "No dataset loaded"}

        try:
            info = {
                "shape": self.current_dataset.shape,
                "columns": list(self.current_dataset.columns),
                "dtypes": self.current_dataset.dtypes.astype(str).to_dict(),
                "missing_values": self.current_dataset.isna().sum().to_dict(),
                "numeric_summary": self.current_dataset.describe().to_dict(),
                "memory_usage": self.current_dataset.memory_usage(deep=True).sum(),
                "timestamp": datetime.now().isoformat()
            }

            return {
                "success": True,
                "info": info
            }
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_column_info(self, column: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific column

        Parameters:
        - column: Column name

        Returns:
        - Dictionary with column information
        """
        if self.current_dataset is None:
            return {"success": False, "error": "No dataset loaded"}

        if column not in self.current_dataset.columns:
            return {"success": False, "error": f"Column {column} not found in dataset"}

        try:
            series = self.current_dataset[column]
            info = {
                "name": column,
                "dtype": str(series.dtype),
                "unique_values": series.nunique(),
                "missing_values": series.isna().sum(),
                "memory_usage": series.memory_usage(deep=True),
            }

            if pd.api.types.is_numeric_dtype(series):
                info.update({
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "25%": series.quantile(0.25),
                    "50%": series.quantile(0.50),
                    "75%": series.quantile(0.75),
                    "max": series.max()
                })
            elif pd.api.types.is_string_dtype(series):
                info.update({
                    "min_length": series.str.len().min(),
                    "max_length": series.str.len().max(),
                    "empty_strings": (series == "").sum(),
                    "value_counts": series.value_counts().head().to_dict()
                })

            return {
                "success": True,
                "info": info
            }
        except Exception as e:
            self.logger.error(f"Error getting column info: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status

        Returns:
        - Dictionary with agent status information
        """
        return {
            "status": self.status,
            "has_data": self.current_dataset is not None,
            "dataset_shape": self.current_dataset.shape if self.current_dataset is not None else None,
            "operation_count": len(self.operation_history),
            "supported_operations": list(self.supported_operations.keys()),
            "supported_transformations": list(self.supported_transformations.keys())
        }

    def reset(self) -> Dict[str, Any]:
        """
        Reset the data engineer's state

        Returns:
        - Dictionary with operation result
        """
        try:
            self.current_dataset = None
            self.operation_history = []
            return {
                "success": True,
                "message": "Data engineer reset successfully"
            }
        except Exception as e:
            self.logger.error(f"Error resetting data engineer: {str(e)}")
            return {"success": False, "error": str(e)}

    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure the data engineer agent

        Parameters:
        - config: Configuration dictionary

        Returns:
        - Dictionary with operation result
        """
        try:
            # Process configuration updates here
            if "status" in config:
                self.status = config["status"]

            return {
                "success": True,
                "message": "Data engineer configured successfully"
            }
        except Exception as e:
            self.logger.error(f"Error configuring data engineer: {str(e)}")
            return {"success": False, "error": str(e)}