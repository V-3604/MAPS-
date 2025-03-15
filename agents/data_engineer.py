import pandas as pd
import numpy as np
import os
import traceback


class DataEngineerAgent:
    def __init__(self, memory_system, function_registry):
        self.memory = memory_system
        self.function_registry = function_registry
        self.current_df = None
        self.dataframes = {}  # Store multiple dataframes if needed

    def process_query(self, query, conversation_manager=None):
        """Process a data engineering query"""
        try:
            # Log the query if conversation manager is provided
            if conversation_manager:
                conversation_manager.add_message("user", query)

            # Basic query parsing - in a real system, you'd use NLP here
            query_lower = query.lower()

            # Handle data loading
            if "load" in query_lower or "read" in query_lower or "import" in query_lower:
                return self._handle_data_loading(query)

            # Handle data cleaning
            if "clean" in query_lower or "missing" in query_lower or "na" in query_lower:
                return self._handle_data_cleaning(query)

            # Handle data transformation
            if "transform" in query_lower or "create" in query_lower or "calculate" in query_lower:
                return self._handle_data_transformation(query)

            # Handle data exploration
            if "explore" in query_lower or "describe" in query_lower or "summary" in query_lower:
                return self._handle_data_exploration(query)

            # Default response if no specific handler matches
            return {
                "success": False,
                "message": "I couldn't understand how to process your data query. Could you be more specific about what data operation you'd like to perform?",
                "result": None
            }

        except Exception as e:
            error_trace = traceback.format_exc()
            error_message = f"Error processing query: {str(e)}\n{error_trace}"

            # Log the error if conversation manager is provided
            if conversation_manager:
                conversation_manager.add_message("system", error_message, {"type": "error"})

            return {
                "success": False,
                "message": f"An error occurred: {str(e)}",
                "error": error_trace,
                "result": None
            }

    def _handle_data_loading(self, query):
        """Handle queries related to loading data"""
        # Simple parsing for demonstration purposes
        file_path = None

        # Extract file path from query
        import re
        file_paths = re.findall(r"['\"](.*?)['\"]", query)
        if file_paths:
            file_path = file_paths[0]
        else:
            # Look for words that might be file paths
            words = query.split()
            for word in words:
                if "." in word and not word.startswith(("http:", "https:")):
                    file_path = word
                    break

        if not file_path:
            return {
                "success": False,
                "message": "Could not determine file path from query. Please specify the file path in quotes.",
                "result": None
            }

        # Ensure the file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"File not found: {file_path}. Please check the file path.",
                "result": None
            }

        # Determine file type and load accordingly
        if file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                self.current_df = df
                self.dataframes["main"] = df

                # Update memory with new data state
                self.memory.update_data_state(file_path, df)
                self.memory.add_operation("Load CSV file", f"pd.read_csv('{file_path}')")
                self.memory.add_key_variable("df", "Main dataframe with loaded data")

                return {
                    "success": True,
                    "message": f"Successfully loaded CSV from {file_path}. Shape: {df.shape}",
                    "result": {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "head": df.head(5).to_dict(orient="records")
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error loading CSV file: {str(e)}",
                    "result": None
                }

        elif file_path.endswith((".xls", ".xlsx")):
            try:
                df = pd.read_excel(file_path)
                self.current_df = df
                self.dataframes["main"] = df

                # Update memory with new data state
                self.memory.update_data_state(file_path, df)
                self.memory.add_operation("Load Excel file", f"pd.read_excel('{file_path}')")
                self.memory.add_key_variable("df", "Main dataframe with loaded data")

                return {
                    "success": True,
                    "message": f"Successfully loaded Excel file from {file_path}. Shape: {df.shape}",
                    "result": {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "head": df.head(5).to_dict(orient="records")
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error loading Excel file: {str(e)}",
                    "result": None
                }
        else:
            return {
                "success": False,
                "message": f"Unsupported file format for {file_path}. Currently supporting .csv, .xls, and .xlsx files.",
                "result": None
            }

    def _handle_data_cleaning(self, query):
        """Handle queries related to data cleaning"""
        if self.current_df is None:
            return {
                "success": False,
                "message": "No data has been loaded yet. Please load a dataset first.",
                "result": None
            }

        # Check for missing values
        if "missing" in query.lower() or "na" in query.lower():
            missing_counts = self.current_df.isna().sum()
            missing_percent = (missing_counts / len(self.current_df)) * 100

            columns_with_missing = missing_counts[missing_counts > 0]

            if len(columns_with_missing) == 0:
                return {
                    "success": True,
                    "message": "No missing values found in the dataset.",
                    "result": {"missing_values": {}}
                }

            missing_info = {col: {"count": int(missing_counts[col]), "percent": float(missing_percent[col])}
                            for col in columns_with_missing.index}

            self.memory.add_operation("Check missing values", "df.isna().sum()", missing_info)

            return {
                "success": True,
                "message": f"Found missing values in {len(columns_with_missing)} columns.",
                "result": {"missing_values": missing_info}
            }

        # Handle drop NA rows
        if "drop" in query.lower() and ("na" in query.lower() or "missing" in query.lower()):
            original_shape = self.current_df.shape

            # Extract column names if specified
            import re
            columns_match = re.search(r"columns\s*[=:]\s*\[(.*?)\]", query, re.IGNORECASE)
            columns_match2 = re.search(r"columns\s+(.*?)(?:\s|$)", query, re.IGNORECASE)

            if columns_match:
                columns_str = columns_match.group(1)
                columns = [col.strip().strip("'\"") for col in columns_str.split(",")]
                self.current_df = self.current_df.dropna(subset=columns)
                operation_code = f"df.dropna(subset={columns})"
            elif columns_match2:
                column_name = columns_match2.group(1).strip().strip("'\"")
                columns = [column_name]
                self.current_df = self.current_df.dropna(subset=columns)
                operation_code = f"df.dropna(subset=['{column_name}'])"
            else:
                self.current_df = self.current_df.dropna()
                operation_code = "df.dropna()"
                columns = []

            new_shape = self.current_df.shape
            rows_removed = original_shape[0] - new_shape[0]

            # Update dataframe in storage
            self.dataframes["main"] = self.current_df

            # Update memory
            self.memory.update_data_state(df=self.current_df)
            self.memory.add_operation(
                f"Drop rows with missing values{' in columns: ' + ', '.join(columns) if columns else ''}",
                operation_code,
                f"Removed {rows_removed} rows. New shape: {new_shape}"
            )

            # Save a checkpoint
            checkpoint_path = self.memory.save_dataframe(self.current_df, "df_after_dropna.pkl")

            return {
                "success": True,
                "message": f"Dropped rows with missing values{' in columns: ' + ', '.join(columns) if columns else ''}. Removed {rows_removed} rows. New shape: {new_shape}",
                "result": {
                    "original_shape": original_shape,
                    "new_shape": new_shape,
                    "rows_removed": rows_removed,
                    "checkpoint_path": checkpoint_path
                }
            }

        # Handle fill NA values
        if "fill" in query.lower() and ("na" in query.lower() or "missing" in query.lower()):
            original_missing = self.current_df.isna().sum().sum()

            # Extract fill value
            import re
            value_match = re.search(r"value\s*[=:]\s*(\S+)", query, re.IGNORECASE)
            column_match = re.search(r"column\s*[=:]\s*['\"](.*?)['\"]", query, re.IGNORECASE)

            fill_value = 0  # Default
            if value_match:
                fill_value_str = value_match.group(1).strip().strip("'\"")
                try:
                    fill_value = int(fill_value_str)
                except ValueError:
                    try:
                        fill_value = float(fill_value_str)
                    except ValueError:
                        fill_value = fill_value_str

            if column_match:
                column = column_match.group(1)
                if column in self.current_df.columns:
                    self.current_df[column] = self.current_df[column].fillna(fill_value)
                    operation_code = f"df['{column}'] = df['{column}'].fillna({fill_value})"
                else:
                    return {
                        "success": False,
                        "message": f"Column '{column}' not found in the dataframe.",
                        "result": None
                    }
            else:
                self.current_df = self.current_df.fillna(fill_value)
                operation_code = f"df.fillna({fill_value})"

            new_missing = self.current_df.isna().sum().sum()
            values_filled = original_missing - new_missing

            # Update dataframe in storage
            self.dataframes["main"] = self.current_df

            # Update memory
            self.memory.update_data_state(df=self.current_df)
            self.memory.add_operation(
                f"Fill missing values with {fill_value}" +
                (f" in column '{column}'" if 'column' in locals() else ""),
                operation_code,
                f"Filled {values_filled} missing values"
            )

            return {
                "success": True,
                "message": f"Filled missing values with {fill_value}" +
                           (f" in column '{column}'" if 'column' in locals() else "") +
                           f". Filled {values_filled} missing values.",
                "result": {
                    "original_missing": int(original_missing),
                    "new_missing": int(new_missing),
                    "values_filled": int(values_filled)
                }
            }

        # Default response if no specific cleaning operation matched
        return {
            "success": False,
            "message": "I couldn't determine what cleaning operation to perform. Please specify if you want to drop or fill missing values.",
            "result": None
        }

    def _handle_data_transformation(self, query):
        """Handle queries related to data transformation"""
        if self.current_df is None:
            return {
                "success": False,
                "message": "No data has been loaded yet. Please load a dataset first.",
                "result": None
            }

        # Handle creating a new column
        if "create" in query.lower() or "new column" in query.lower():
            import re

            # Try different patterns to extract column name and expression
            col_patterns = [
                r"column\s+['\"](.*?)['\"]",  # column 'name'
                r"column\s+(\w+)",  # column name
                r"['\"](.*?)['\"].*?=",  # 'name' =
                r"(\w+)\s+as\s+",  # name as
                r"(\w+)\s+(equals|=)"  # name equals
            ]

            expr_patterns = [
                r"=\s*(.*?)(?:$|where|with)",  # = expression
                r"as\s+(.*?)(?:$|where|with)",  # as expression
                r"equals\s+(.*?)(?:$|where|with)"  # equals expression
            ]

            # Extract column name
            new_col = None
            for pattern in col_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match and match.group(1):
                    new_col = match.group(1).strip().strip("'\"")
                    break

            # Extract expression
            expr_str = None
            for pattern in expr_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match and match.group(1):
                    expr_str = match.group(1).strip()
                    break

            # If we couldn't find a column name or expression
            if not new_col:
                return {
                    "success": False,
                    "message": "Could not determine new column name. Please specify 'column NAME' in your query.",
                    "result": None
                }

            if not expr_str:
                return {
                    "success": False,
                    "message": "Could not determine the expression for the new column. Please use 'column NAME = EXPRESSION' format.",
                    "result": None
                }

            # Security check - very basic, would need more sophisticated validation in production
            forbidden_terms = ["import", "exec", "eval", "system", "os.", "subprocess"]
            if any(term in expr_str for term in forbidden_terms):
                return {
                    "success": False,
                    "message": "Expression contains potentially unsafe operations. Please use only basic arithmetic and column references.",
                    "result": None
                }

            try:
                # Convert text descriptions to actual operations
                expr_str = expr_str.replace("times", "*").replace("divided by", "/").replace("plus", "+").replace(
                    "minus", "-")

                # Replace column references
                for col in self.current_df.columns:
                    quoted_col = f"'{col}'"
                    if quoted_col in expr_str:
                        expr_str = expr_str.replace(quoted_col, f"self.current_df['{col}']")
                    elif f'"{col}"' in expr_str:
                        expr_str = expr_str.replace(f'"{col}"', f"self.current_df['{col}']")
                    elif col in expr_str:
                        # Only replace if it's a whole word to avoid partial matches
                        expr_str = re.sub(r'\b' + col + r'\b', f"self.current_df['{col}']", expr_str)

                # Execute the expression
                print(f"Executing expression: {expr_str}")  # Debug info
                result = eval(expr_str)
                self.current_df[new_col] = result

                # Update dataframe in storage
                self.dataframes["main"] = self.current_df

                # Create a more readable version of the code for storing in memory
                readable_expr = expr_str.replace("self.current_df", "df")
                operation_code = f"df['{new_col}'] = {readable_expr}"

                # Update memory
                self.memory.update_data_state(df=self.current_df)
                self.memory.add_operation(
                    f"Create new column '{new_col}'",
                    operation_code,
                    f"Added column '{new_col}' with data type {result.dtype}"
                )

                return {
                    "success": True,
                    "message": f"Created new column '{new_col}' with expression: {expr_str}",
                    "result": {
                        "column_name": new_col,
                        "dtype": str(result.dtype),
                        "sample": result.head(5).tolist() if hasattr(result, 'head') else str(result)[:100]
                    }
                }

            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error creating new column: {str(e)}",
                    "result": None
                }

    def _handle_data_exploration(self, query):
        """Handle queries related to data exploration"""
        if self.current_df is None:
            return {
                "success": False,
                "message": "No data has been loaded yet. Please load a dataset first.",
                "result": None
            }

        # Handle describe operation
        if "describe" in query.lower() or "summary" in query.lower() or "statistics" in query.lower():
            try:
                description = self.current_df.describe(include='all')

                # Update memory
                self.memory.add_operation(
                    "Get descriptive statistics",
                    "df.describe(include='all')",
                    "Generated statistical summary"
                )

                return {
                    "success": True,
                    "message": "Generated descriptive statistics for the dataframe",
                    "result": {
                        "description": description.to_dict(),
                        "shape": self.current_df.shape,
                        "dtypes": {col: str(dtype) for col, dtype in self.current_df.dtypes.items()}
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error generating descriptive statistics: {str(e)}",
                    "result": None
                }

        # Handle info about columns
        if "info" in query.lower() or "columns" in query.lower() or "dtypes" in query.lower():
            try:
                # Get column information
                info = {
                    "shape": self.current_df.shape,
                    "columns": list(self.current_df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in self.current_df.dtypes.items()},
                    "missing": {col: int(self.current_df[col].isna().sum()) for col in self.current_df.columns},
                    "unique_counts": {col: int(self.current_df[col].nunique()) for col in self.current_df.columns}
                }

                # Update memory
                self.memory.add_operation(
                    "Get column information",
                    "df.info() equivalent",
                    "Generated column information"
                )

                return {
                    "success": True,
                    "message": f"Retrieved information about {len(self.current_df.columns)} columns",
                    "result": info
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error retrieving column information: {str(e)}",
                    "result": None
                }

        # Handle data sample request - expanded pattern matching for this common request
        if any(term in query.lower() for term in ["sample", "head", "show", "first", "top", "display", "view"]) and any(
                term in query.lower() for term in ["rows", "data", "entries", "records"]):
            try:
                # Extract number of rows if specified
                import re
                rows_match = re.search(r"(\d+)\s+rows", query, re.IGNORECASE)

                n_rows = 5  # Default
                if rows_match:
                    n_rows = int(rows_match.group(1))

                sample = self.current_df.head(n_rows)

                # Update memory
                self.memory.add_operation(
                    f"Show sample of {n_rows} rows",
                    f"df.head({n_rows})",
                    f"Retrieved {n_rows} sample rows"
                )

                return {
                    "success": True,
                    "message": f"Retrieved {n_rows} sample rows from the dataframe",
                    "result": {
                        "sample": sample.to_dict(orient="records"),
                        "shape": self.current_df.shape
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error retrieving data sample: {str(e)}",
                    "result": None
                }

        # Default response for exploration
        return {
            "success": False,
            "message": "I couldn't determine what exploration operation to perform. Try asking for 'describe', 'info', or 'sample'.",
            "result": None
        }

    def execute_code(self, code, session_id=None):
        """Execute arbitrary pandas code with safety checks"""
        # Security check - very basic, would need more sophisticated validation in production
        forbidden_terms = ["import", "exec", "eval", "system", "os.", "subprocess"]
        if any(term in code for term in forbidden_terms):
            return {
                "success": False,
                "message": "Code contains potentially unsafe operations. Please use only pandas operations.",
                "result": None
            }

        try:
            # Add safety by limiting the namespace
            namespace = {
                "pd": pd,
                "np": np,
                "df": self.current_df
            }

            # Execute code
            result = exec(code, namespace)

            # Check if df was modified
            if "df" in namespace and id(namespace["df"]) != id(self.current_df):
                self.current_df = namespace["df"]
                self.dataframes["main"] = self.current_df

                # Update memory
                self.memory.update_data_state(df=self.current_df)
                self.memory.add_operation(
                    "Execute custom code",
                    code,
                    f"Modified dataframe. New shape: {self.current_df.shape}"
                )

            return {
                "success": True,
                "message": "Code executed successfully",
                "result": {
                    "output": str(result) if result is not None else "No output",
                    "df_updated": id(namespace["df"]) != id(self.current_df),
                    "current_shape": self.current_df.shape if self.current_df is not None else None
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing code: {str(e)}",
                "result": None
            }