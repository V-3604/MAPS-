# utils/data_validation.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class DataValidator:
    def __init__(self):
        self.validation_rules = {
            "numeric": self._validate_numeric,
            "range": self._validate_range,
            "categorical": self._validate_categorical,
            "datetime": self._validate_datetime,
            "missing": self._validate_missing,
            "unique": self._validate_unique,
            "pattern": self._validate_pattern,
            "correlation": self._validate_correlation,
            "custom": self._validate_custom
        }

    def validate_dataset(self,
                         df: pd.DataFrame,
                         rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a dataset against a set of rules

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to validate
        rules : Dict[str, Dict[str, Any]]
            Validation rules structure: {column_name: {rule_type: rule_params}}

        Returns:
        --------
        Dict[str, Any]
            Validation results with errors and warnings
        """
        if df is None:
            return {
                "valid": False,
                "errors": ["No dataset provided for validation"],
                "warnings": [],
                "summary": {"error_count": 1, "warning_count": 0}
            }

        errors = []
        warnings = []
        column_results = {}

        # Validate column existence
        missing_columns = [col for col in rules.keys() if col not in df.columns]
        if missing_columns:
            errors.append(f"Columns not found in dataset: {', '.join(missing_columns)}")

        # Apply validation rules to each column
        for column, column_rules in rules.items():
            if column in df.columns:
                column_results[column] = self._validate_column(df, column, column_rules)
                errors.extend(column_results[column]["errors"])
                warnings.extend(column_results[column]["warnings"])

        # Determine if the dataset is valid (no errors)
        is_valid = len(errors) == 0

        # Create summary statistics
        summary = {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "column_count": len(rules),
            "columns_validated": len(column_results),
            "timestamp": datetime.now().isoformat()
        }

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "column_results": column_results,
            "summary": summary
        }

    def _validate_column(self,
                         df: pd.DataFrame,
                         column: str,
                         rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single column against its rules

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe containing the column
        column : str
            The column to validate
        rules : Dict[str, Any]
            Rules to apply to the column

        Returns:
        --------
        Dict[str, Any]
            Column validation results
        """
        column_errors = []
        column_warnings = []
        rule_results = {}

        for rule_type, rule_params in rules.items():
            if rule_type in self.validation_rules:
                result = self.validation_rules[rule_type](df, column, rule_params)
                rule_results[rule_type] = result

                if not result["valid"]:
                    if result.get("level", "error") == "error":
                        column_errors.append(result["message"])
                    else:
                        column_warnings.append(result["message"])
            else:
                column_warnings.append(f"Unknown validation rule type: {rule_type}")

        return {
            "valid": len(column_errors) == 0,
            "errors": column_errors,
            "warnings": column_warnings,
            "rule_results": rule_results
        }

    def _validate_numeric(self,
                          df: pd.DataFrame,
                          column: str,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that a column contains numeric values"""
        try:
            data = df[column]
            is_numeric = pd.api.types.is_numeric_dtype(data)

            if not is_numeric:
                # Check if values could be converted to numeric
                try:
                    pd.to_numeric(data)
                    return {
                        "valid": True,
                        "level": "warning",
                        "message": f"Column '{column}' contains values that could be converted to numeric"
                    }
                except:
                    return {
                        "valid": False,
                        "message": f"Column '{column}' should contain numeric values"
                    }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating numeric values in '{column}': {str(e)}"
            }

    def _validate_range(self,
                        df: pd.DataFrame,
                        column: str,
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that numeric values are within a specified range"""
        try:
            data = df[column]

            # Ensure column is numeric
            if not pd.api.types.is_numeric_dtype(data):
                return {
                    "valid": False,
                    "message": f"Range validation requires numeric data in '{column}'"
                }

            min_val = params.get("min")
            max_val = params.get("max")

            # Check minimum value if specified
            if min_val is not None:
                below_min = data < min_val
                if below_min.any():
                    count = below_min.sum()
                    return {
                        "valid": False,
                        "message": f"Column '{column}' has {count} values below minimum {min_val}"
                    }

            # Check maximum value if specified
            if max_val is not None:
                above_max = data > max_val
                if above_max.any():
                    count = above_max.sum()
                    return {
                        "valid": False,
                        "message": f"Column '{column}' has {count} values above maximum {max_val}"
                    }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating range in '{column}': {str(e)}"
            }

    def _validate_categorical(self,
                              df: pd.DataFrame,
                              column: str,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that values belong to a specified set of categories"""
        try:
            data = df[column]
            categories = params.get("categories", [])

            if not categories:
                return {
                    "valid": False,
                    "message": f"No categories specified for '{column}'"
                }

            invalid_values = data[~data.isin(categories)].unique()

            if len(invalid_values) > 0:
                return {
                    "valid": False,
                    "message": f"Column '{column}' contains {len(invalid_values)} invalid categories: {invalid_values[:5]}"
                }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating categories in '{column}': {str(e)}"
            }

    def _validate_datetime(self,
                           df: pd.DataFrame,
                           column: str,
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that values can be parsed as datetimes"""
        try:
            data = df[column]

            # Check if already datetime dtype
            if pd.api.types.is_datetime64_dtype(data):
                valid = True
            else:
                # Try to convert to datetime
                try:
                    format_str = params.get("format")
                    if format_str:
                        pd.to_datetime(data, format=format_str)
                    else:
                        pd.to_datetime(data)
                    valid = True
                except:
                    valid = False

            if not valid:
                return {
                    "valid": False,
                    "message": f"Column '{column}' contains values that cannot be parsed as datetime"
                }

            # Check date range if specified
            min_date = params.get("min_date")
            max_date = params.get("max_date")

            if valid and (min_date or max_date):
                date_data = pd.to_datetime(data)

                if min_date:
                    min_date = pd.to_datetime(min_date)
                    below_min = date_data < min_date
                    if below_min.any():
                        return {
                            "valid": False,
                            "message": f"Column '{column}' has {below_min.sum()} dates before {min_date}"
                        }

                if max_date:
                    max_date = pd.to_datetime(max_date)
                    above_max = date_data > max_date
                    if above_max.any():
                        return {
                            "valid": False,
                            "message": f"Column '{column}' has {above_max.sum()} dates after {max_date}"
                        }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating datetime in '{column}': {str(e)}"
            }

    def _validate_missing(self,
                          df: pd.DataFrame,
                          column: str,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that missing values are within acceptable threshold"""
        try:
            data = df[column]
            missing_count = data.isna().sum()
            total_count = len(data)
            missing_percent = missing_count / total_count if total_count > 0 else 0

            threshold = params.get("threshold", 0)

            if missing_percent > threshold:
                return {
                    "valid": False,
                    "message": f"Column '{column}' has {missing_count} missing values ({missing_percent:.2%}), exceeding threshold of {threshold:.2%}"
                }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating missing values in '{column}': {str(e)}"
            }

    def _validate_unique(self,
                         df: pd.DataFrame,
                         column: str,
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that values in a column are unique"""
        try:
            data = df[column]
            unique_count = data.nunique()
            total_count = len(data)

            if params.get("is_primary_key", False):
                # Must be completely unique
                if unique_count < total_count:
                    duplicates = data[data.duplicated()].unique()
                    dup_sample = str(duplicates[:3].tolist())
                    return {
                        "valid": False,
                        "message": f"Column '{column}' has duplicate values, including: {dup_sample}"
                    }
            else:
                # Check for minimum unique percentage
                min_unique_percent = params.get("min_unique_percent", 0)
                actual_percent = unique_count / total_count if total_count > 0 else 0

                if actual_percent < min_unique_percent:
                    return {
                        "valid": False,
                        "message": f"Column '{column}' has only {unique_count} unique values ({actual_percent:.2%}), below threshold of {min_unique_percent:.2%}"
                    }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating uniqueness in '{column}': {str(e)}"
            }

    def _validate_pattern(self,
                          df: pd.DataFrame,
                          column: str,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that values match a regex pattern"""
        try:
            import re
            data = df[column].astype(str)
            pattern = params.get("regex", "")

            if not pattern:
                return {
                    "valid": False,
                    "message": f"No regex pattern specified for '{column}'"
                }

            # Compile pattern for efficiency
            regex = re.compile(pattern)

            # Check which values don't match the pattern
            non_matching = data[~data.str.match(regex)]

            if len(non_matching) > 0:
                sample = non_matching.head(3).tolist()
                return {
                    "valid": False,
                    "message": f"Column '{column}' has {len(non_matching)} values not matching pattern, including: {sample}"
                }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating pattern in '{column}': {str(e)}"
            }

    def _validate_correlation(self,
                              df: pd.DataFrame,
                              column: str,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate correlation between columns"""
        try:
            if not pd.api.types.is_numeric_dtype(df[column]):
                return {
                    "valid": False,
                    "message": f"Correlation validation requires numeric data in '{column}'"
                }

            target_column = params.get("target_column")
            min_correlation = params.get("min_correlation", 0)
            max_correlation = params.get("max_correlation", 1)

            if not target_column or target_column not in df.columns:
                return {
                    "valid": False,
                    "message": f"Invalid target column for correlation with '{column}'"
                }

            if not pd.api.types.is_numeric_dtype(df[target_column]):
                return {
                    "valid": False,
                    "message": f"Correlation target column '{target_column}' must be numeric"
                }

            # Calculate correlation
            correlation = df[[column, target_column]].corr().iloc[0, 1]

            if correlation < min_correlation:
                return {
                    "valid": False,
                    "message": f"Correlation between '{column}' and '{target_column}' is {correlation:.2f}, below minimum {min_correlation}"
                }

            if correlation > max_correlation:
                return {
                    "valid": False,
                    "message": f"Correlation between '{column}' and '{target_column}' is {correlation:.2f}, above maximum {max_correlation}"
                }

            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating correlation for '{column}': {str(e)}"
            }

    def _validate_custom(self,
                         df: pd.DataFrame,
                         column: str,
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a custom validation function"""
        try:
            validation_func = params.get("function")

            if not validation_func or not callable(validation_func):
                return {
                    "valid": False,
                    "message": f"No valid function provided for custom validation of '{column}'"
                }

            # Apply the custom function
            result = validation_func(df, column)

            # Function should return a dict with at least {"valid": bool, "message": str}
            if not isinstance(result, dict) or "valid" not in result:
                return {
                    "valid": False,
                    "message": f"Custom validation function for '{column}' returned invalid result"
                }

            return result
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error in custom validation for '{column}': {str(e)}"
            }