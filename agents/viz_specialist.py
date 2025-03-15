import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import traceback
from datetime import datetime


class VizSpecialistAgent:
    def __init__(self, memory_system, function_registry, output_dir="./data/visualizations/"):
        self.memory = memory_system
        self.function_registry = function_registry
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def process_query(self, query, df, conversation_manager=None):
        """Process a visualization query"""
        try:
            # Log the query if conversation manager is provided
            if conversation_manager:
                conversation_manager.add_message("user", query)

            # Check if we have a dataframe to work with
            if df is None or df.empty:
                return {
                    "success": False,
                    "message": "No data available for visualization. Please load or process data first.",
                    "result": None
                }

            # Basic query parsing - in a real system, you'd use NLP here
            query_lower = query.lower()

            # Histogram
            if "histogram" in query_lower or "distribution" in query_lower:
                return self._create_histogram(query, df)

            # Scatter plot
            if "scatter" in query_lower or "relationship" in query_lower or "correlation" in query_lower:
                return self._create_scatter(query, df)

            # Bar plot
            if "bar" in query_lower or "barplot" in query_lower or "count" in query_lower:
                return self._create_barplot(query, df)

            # Box plot
            if "box" in query_lower or "boxplot" in query_lower:
                return self._create_boxplot(query, df)

            # Heatmap
            if "heatmap" in query_lower or "correlation matrix" in query_lower:
                return self._create_heatmap(query, df)

            # Line plot
            if "line" in query_lower or "trend" in query_lower or "time series" in query_lower:
                return self._create_lineplot(query, df)

            # Suggest visualizations
            if "suggest" in query_lower or "recommend" in query_lower:
                return self._suggest_visualizations(query, df)

            # Default response if no specific handler matches
            return {
                "success": False,
                "message": "I couldn't determine what visualization to create. Please specify a visualization type like histogram, scatter plot, bar plot, etc.",
                "result": None
            }

        except Exception as e:
            error_trace = traceback.format_exc()
            error_message = f"Error processing visualization query: {str(e)}\n{error_trace}"

            # Log the error if conversation manager is provided
            if conversation_manager:
                conversation_manager.add_message("system", error_message, {"type": "error"})

            return {
                "success": False,
                "message": f"An error occurred: {str(e)}",
                "error": error_trace,
                "result": None
            }

    def _extract_columns(self, query, df):
        """Extract column names from query"""
        import re

        # All columns present in the dataframe
        all_columns = df.columns.tolist()

        # Extract columns enclosed in quotes
        quoted_columns = re.findall(r"['\"]([^'\"]+)['\"]", query)

        # Look for column names specified without quotes
        unquoted_columns = []
        words = query.split()
        for word in words:
            # Clean the word from punctuation
            clean_word = word.strip(",.;:'\"()")
            if clean_word in all_columns and clean_word not in quoted_columns:
                unquoted_columns.append(clean_word)

        # Combine and deduplicate
        columns = quoted_columns + unquoted_columns
        return list(dict.fromkeys(columns))  # Remove duplicates while preserving order

    def _generate_filename(self, viz_type):
        """Generate a unique filename for the visualization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{viz_type}_{timestamp}.png"

    def _create_histogram(self, query, df):
        """Create a histogram visualization"""
        columns = self._extract_columns(query, df)

        if not columns:
            # Try to find numeric columns if none specified
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                columns = [numeric_cols[0]]  # Use the first numeric column
            else:
                return {
                    "success": False,
                    "message": "Could not determine which column to plot. Please specify a column name in your query.",
                    "result": None
                }

        # Only use the first column mentioned for histogram
        column = columns[0]

        # Check if column exists
        if column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{column}' not found in the dataframe.",
                "result": None
            }

        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "success": False,
                "message": f"Column '{column}' is not numeric. Histogram requires numeric data.",
                "result": None
            }

        try:
            # Create the plot
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=column, kde=True)
            plt.title(f"Distribution of {column}")
            plt.tight_layout()

            # Save the plot
            filename = self._generate_filename("histogram")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            # Update memory
            viz_info = self.memory.add_visualization(
                "histogram",
                f"Distribution of {column}",
                [column],
                filepath
            )

            # Add operation to memory
            self.memory.add_operation(
                f"Create histogram for column '{column}'",
                f"sns.histplot(data=df, x='{column}', kde=True)",
                f"Created histogram visualization saved to {filename}"
            )

            return {
                "success": True,
                "message": f"Created histogram for column '{column}'",
                "result": {
                    "filepath": filepath,
                    "column": column,
                    "viz_info": viz_info
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating histogram: {str(e)}",
                "result": None
            }

    def _create_scatter(self, query, df):
        """Create a scatter plot visualization"""
        columns = self._extract_columns(query, df)

        if len(columns) < 2:
            # Try to find numeric columns if not enough specified
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 2:
                columns = numeric_cols[:2]  # Use the first two numeric columns
            else:
                return {
                    "success": False,
                    "message": "Could not determine which columns to plot. Please specify at least two column names for a scatter plot.",
                    "result": None
                }

        # Use the first two columns mentioned for x and y
        x_col = columns[0]
        y_col = columns[1]

        # Check if columns exist
        for col in [x_col, y_col]:
            if col not in df.columns:
                return {
                    "success": False,
                    "message": f"Column '{col}' not found in the dataframe.",
                    "result": None
                }

        # Check if columns are numeric
        for col in [x_col, y_col]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {
                    "success": False,
                    "message": f"Column '{col}' is not numeric. Scatter plot requires numeric data.",
                    "result": None
                }

        # Check for optional hue parameter
        hue_col = None
        if len(columns) > 2:
            hue_col = columns[2]
            if hue_col not in df.columns:
                hue_col = None

        try:
            # Create the plot
            plt.figure(figsize=(10, 6))
            if hue_col:
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
                plt.title(f"Relationship between {x_col} and {y_col} (colored by {hue_col})")
            else:
                sns.scatterplot(data=df, x=x_col, y=y_col)
                plt.title(f"Relationship between {x_col} and {y_col}")

            plt.tight_layout()

            # Save the plot
            filename = self._generate_filename("scatter")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            # Update memory
            columns_used = [x_col, y_col]
            if hue_col:
                columns_used.append(hue_col)

            viz_info = self.memory.add_visualization(
                "scatter",
                f"Relationship between {x_col} and {y_col}" + (f" (colored by {hue_col})" if hue_col else ""),
                columns_used,
                filepath
            )

            # Add operation to memory
            code_str = f"sns.scatterplot(data=df, x='{x_col}', y='{y_col}'" + (
                f", hue='{hue_col}')" if hue_col else ")")
            self.memory.add_operation(
                f"Create scatter plot for {x_col} vs {y_col}" + (f" with hue={hue_col}" if hue_col else ""),
                code_str,
                f"Created scatter plot visualization saved to {filename}"
            )

            return {
                "success": True,
                "message": f"Created scatter plot for {x_col} vs {y_col}" + (f" with hue={hue_col}" if hue_col else ""),
                "result": {
                    "filepath": filepath,
                    "x_column": x_col,
                    "y_column": y_col,
                    "hue_column": hue_col,
                    "viz_info": viz_info
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating scatter plot: {str(e)}",
                "result": None
            }

    def _create_barplot(self, query, df):
        """Create a bar plot visualization"""
        columns = self._extract_columns(query, df)

        if len(columns) < 1:
            # Try to find categorical columns if none specified
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if categorical_cols and numeric_cols:
                columns = [categorical_cols[0], numeric_cols[0]]  # Use first categorical and numeric columns
            elif categorical_cols:
                columns = [categorical_cols[0]]  # Use just the categorical column
            elif numeric_cols:
                # Find a numeric column with few unique values
                for col in numeric_cols:
                    if df[col].nunique() < 20:
                        columns = [col]
                        break
                if not columns:
                    columns = [numeric_cols[0]]  # Fallback to first numeric
            else:
                return {
                    "success": False,
                    "message": "Could not determine which columns to plot. Please specify at least one column name for a bar plot.",
                    "result": None
                }

        # Determine x and y columns based on data types
        x_col = columns[0]
        y_col = None

        # Check if column exists
        if x_col not in df.columns:
            return {
                "success": False,
                "message": f"Column '{x_col}' not found in the dataframe.",
                "result": None
            }

        # If we have a second column, use it as y
        if len(columns) > 1:
            y_col = columns[1]
            if y_col not in df.columns:
                y_col = None

        # If we only have one column, determine if it should be x or y
        if y_col is None:
            # If x is numeric with many values, it's probably better as y
            if pd.api.types.is_numeric_dtype(df[x_col]) and df[x_col].nunique() > 20:
                y_col = x_col

                # Find a good categorical column for x
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    x_col = categorical_cols[0]
                else:
                    # Find a numeric column with few values
                    for col in df.select_dtypes(include=np.number).columns:
                        if col != y_col and df[col].nunique() < 20:
                            x_col = col
                            break

        try:
            # Create the plot
            plt.figure(figsize=(12, 6))

            if y_col:
                # Standard bar plot with x and y
                sns.barplot(data=df, x=x_col, y=y_col)
                plt.title(f"Bar plot of {y_col} by {x_col}")
            else:
                # Count plot (just x)
                sns.countplot(data=df, x=x_col, order=df[x_col].value_counts().index[:20])
                plt.title(f"Count of records by {x_col}")

            # Rotate x-labels if there are many categories
            if df[x_col].nunique() > 5:
                plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            # Save the plot
            filename = self._generate_filename("barplot")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            # Update memory
            columns_used = [x_col]
            if y_col:
                columns_used.append(y_col)

            description = f"Bar plot of {y_col} by {x_col}" if y_col else f"Count of records by {x_col}"
            viz_info = self.memory.add_visualization(
                "barplot" if y_col else "countplot",
                description,
                columns_used,
                filepath
            )

            # Add operation to memory
            if y_col:
                code_str = f"sns.barplot(data=df, x='{x_col}', y='{y_col}')"
            else:
                code_str = f"sns.countplot(data=df, x='{x_col}')"

            self.memory.add_operation(
                f"Create {'bar plot' if y_col else 'count plot'} for {x_col}" + (f" vs {y_col}" if y_col else ""),
                code_str,
                f"Created {'bar' if y_col else 'count'} plot visualization saved to {filename}"
            )

            return {
                "success": True,
                "message": f"Created {'bar plot' if y_col else 'count plot'} for {x_col}" + (
                    f" vs {y_col}" if y_col else ""),
                "result": {
                    "filepath": filepath,
                    "x_column": x_col,
                    "y_column": y_col,
                    "viz_info": viz_info
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating bar plot: {str(e)}",
                "result": None
            }

    def _create_boxplot(self, query, df):
        """Create a box plot visualization"""
        columns = self._extract_columns(query, df)

        if len(columns) < 1:
            # Try to find categorical and numeric columns if none specified
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if categorical_cols and numeric_cols:
                columns = [categorical_cols[0], numeric_cols[0]]  # Use first categorical and numeric columns
            elif numeric_cols:
                columns = [numeric_cols[0]]  # Use just the numeric column
            else:
                return {
                    "success": False,
                    "message": "Could not determine which columns to plot. Please specify at least one numeric column name for a box plot.",
                    "result": None
                }

        # First, find a numeric column for y
        y_col = None
        x_col = None

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                y_col = col
                break

        if y_col is None:
            # Try to find any numeric column
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                y_col = numeric_cols[0]
            else:
                return {
                    "success": False,
                    "message": "No numeric column found for box plot. Box plots require at least one numeric column.",
                    "result": None
                }

        # Now find a categorical column for x (if provided)
        remaining_cols = [col for col in columns if col != y_col and col in df.columns]
        if remaining_cols:
            x_col = remaining_cols[0]
        else:
            # Try to find a good categorical column
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                x_col = categorical_cols[0]

        try:
            # Create the plot
            plt.figure(figsize=(12, 6))

            if x_col:
                sns.boxplot(data=df, x=x_col, y=y_col)
                plt.title(f"Distribution of {y_col} by {x_col}")

                # Rotate x-labels if there are many categories
                if df[x_col].nunique() > 5:
                    plt.xticks(rotation=45, ha='right')
            else:
                sns.boxplot(data=df, y=y_col)
                plt.title(f"Distribution of {y_col}")

            plt.tight_layout()

            # Save the plot
            filename = self._generate_filename("boxplot")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            # Update memory
            columns_used = [y_col]
            if x_col:
                columns_used.append(x_col)

            description = f"Distribution of {y_col}" + (f" by {x_col}" if x_col else "")
            viz_info = self.memory.add_visualization(
                "boxplot",
                description,
                columns_used,
                filepath
            )

            # Add operation to memory
            if x_col:
                code_str = f"sns.boxplot(data=df, x='{x_col}', y='{y_col}')"
            else:
                code_str = f"sns.boxplot(data=df, y='{y_col}')"

            self.memory.add_operation(
                f"Create box plot for {y_col}" + (f" by {x_col}" if x_col else ""),
                code_str,
                f"Created box plot visualization saved to {filename}"
            )

            return {
                "success": True,
                "message": f"Created box plot for {y_col}" + (f" by {x_col}" if x_col else ""),
                "result": {
                    "filepath": filepath,
                    "y_column": y_col,
                    "x_column": x_col,
                    "viz_info": viz_info
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating box plot: {str(e)}",
                "result": None
            }

    def _create_heatmap(self, query, df):
        """Create a correlation heatmap"""
        columns = self._extract_columns(query, df)

        # If specific columns are provided, use only those
        if columns:
            # Check if columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {
                    "success": False,
                    "message": f"Columns not found in the dataframe: {', '.join(missing_cols)}",
                    "result": None
                }

            # Check if columns are numeric
            non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return {
                    "success": False,
                    "message": f"Non-numeric columns can't be used in correlation heatmap: {', '.join(non_numeric)}",
                    "result": None
                }

            data_to_plot = df[columns]
        else:
            # Use all numeric columns
            data_to_plot = df.select_dtypes(include=np.number)

            if data_to_plot.shape[1] < 2:
                return {
                    "success": False,
                    "message": "Not enough numeric columns for a correlation heatmap. Need at least 2 numeric columns.",
                    "result": None
                }

        try:
            # Calculate correlation matrix
            corr = data_to_plot.corr()

            # Create the plot
            plt.figure(figsize=(12, 10))

            mask = np.triu(np.ones_like(corr, dtype=bool))  # Create mask for upper triangle

            # Plot the heatmap
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1, vmax=1,
                linewidths=0.5
            )
            plt.title("Correlation Matrix")
            plt.tight_layout()

            # Save the plot
            filename = self._generate_filename("heatmap")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            # Update memory
            columns_used = data_to_plot.columns.tolist()

            viz_info = self.memory.add_visualization(
                "heatmap",
                "Correlation matrix",
                columns_used,
                filepath
            )

            # Add operation to memory
            code_str = "sns.heatmap(df.corr(), annot=True, cmap='coolwarm')"
            self.memory.add_operation(
                "Create correlation heatmap",
                code_str,
                f"Created correlation heatmap visualization saved to {filename}"
            )

            return {
                "success": True,
                "message": f"Created correlation heatmap using {len(columns_used)} numeric columns",
                "result": {
                    "filepath": filepath,
                    "columns": columns_used,
                    "viz_info": viz_info
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating heatmap: {str(e)}",
                "result": None
            }

    def _create_lineplot(self, query, df):
        """Create a line plot visualization"""
        columns = self._extract_columns(query, df)

        if len(columns) < 2:
            # Try to find appropriate columns if none specified
            date_cols = []
            for col in df.columns:
                # Check if column might be a date
                if pd.api.types.is_datetime64_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
                    date_cols.append(col)

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if date_cols and numeric_cols:
                columns = [date_cols[0], numeric_cols[0]]
            elif numeric_cols and len(numeric_cols) >= 2:
                columns = numeric_cols[:2]
            else:
                return {
                    "success": False,
                    "message": "Could not determine which columns to plot. Please specify at least x and y column names for a line plot.",
                    "result": None
                }

        # Use the first two columns for x and y
        x_col = columns[0]
        y_col = columns[1]

        # Check if columns exist
        for col in [x_col, y_col]:
            if col not in df.columns:
                return {
                    "success": False,
                    "message": f"Column '{col}' not found in the dataframe.",
                    "result": None
                }

        # Check for optional hue parameter
        hue_col = None
        if len(columns) > 2:
            hue_col = columns[2]
            if hue_col not in df.columns:
                hue_col = None

        try:
            # Create a copy of the dataframe for plotting
            plot_df = df.copy()

            # If x_col is a date string, convert to datetime
            if not pd.api.types.is_datetime64_dtype(plot_df[x_col]) and plot_df[x_col].dtype == 'object':
                try:
                    plot_df[x_col] = pd.to_datetime(plot_df[x_col])
                except:
                    pass  # If conversion fails, use as is

            # Create the plot
            plt.figure(figsize=(12, 6))

            if hue_col:
                sns.lineplot(data=plot_df, x=x_col, y=y_col, hue=hue_col)
                plt.title(f"Trend of {y_col} over {x_col} (grouped by {hue_col})")
            else:
                sns.lineplot(data=plot_df, x=x_col, y=y_col)
                plt.title(f"Trend of {y_col} over {x_col}")

            # Rotate x-labels if there are many points
            if plot_df[x_col].nunique() > 10:
                plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            # Save the plot
            filename = self._generate_filename("lineplot")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            # Update memory
            columns_used = [x_col, y_col]
            if hue_col:
                columns_used.append(hue_col)

            viz_info = self.memory.add_visualization(
                "lineplot",
                f"Trend of {y_col} over {x_col}" + (f" (grouped by {hue_col})" if hue_col else ""),
                columns_used,
                filepath
            )

            # Add operation to memory
            code_str = f"sns.lineplot(data=df, x='{x_col}', y='{y_col}'" + (f", hue='{hue_col}')" if hue_col else ")")
            self.memory.add_operation(
                f"Create line plot for {y_col} over {x_col}" + (f" with hue={hue_col}" if hue_col else ""),
                code_str,
                f"Created line plot visualization saved to {filename}"
            )

            return {
                "success": True,
                "message": f"Created line plot for {y_col} over {x_col}" + (f" with hue={hue_col}" if hue_col else ""),
                "result": {
                    "filepath": filepath,
                    "x_column": x_col,
                    "y_column": y_col,
                    "hue_column": hue_col,
                    "viz_info": viz_info
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating line plot: {str(e)}",
                "result": None
            }

    def _suggest_visualizations(self, query, df):
        """Suggest appropriate visualizations for the data"""
        columns = self._extract_columns(query, df)

        if not columns:
            # Suggest based on data types
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = []

            for col in df.columns:
                if pd.api.types.is_datetime64_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
                    date_cols.append(col)

            suggestions = []

            # For numeric columns
            if numeric_cols:
                suggestions.append({
                    "type": "histogram",
                    "description": f"Distribution of {numeric_cols[0]}",
                    "columns": [numeric_cols[0]]
                })

                if len(numeric_cols) >= 2:
                    suggestions.append({
                        "type": "scatter",
                        "description": f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                        "columns": [numeric_cols[0], numeric_cols[1]]
                    })

                    suggestions.append({
                        "type": "heatmap",
                        "description": "Correlation matrix of numeric columns",
                        "columns": numeric_cols
                    })

            # For categorical columns
            if categorical_cols:
                suggestions.append({
                    "type": "countplot",
                    "description": f"Count of records by {categorical_cols[0]}",
                    "columns": [categorical_cols[0]]
                })

                if numeric_cols:
                    suggestions.append({
                        "type": "boxplot",
                        "description": f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}",
                        "columns": [categorical_cols[0], numeric_cols[0]]
                    })

                    suggestions.append({
                        "type": "barplot",
                        "description": f"Average {numeric_cols[0]} by {categorical_cols[0]}",
                        "columns": [categorical_cols[0], numeric_cols[0]]
                    })

            # For time series
            if date_cols and numeric_cols:
                suggestions.append({
                    "type": "lineplot",
                    "description": f"Trend of {numeric_cols[0]} over {date_cols[0]}",
                    "columns": [date_cols[0], numeric_cols[0]]
                })

                if categorical_cols:
                    suggestions.append({
                        "type": "lineplot",
                        "description": f"Trend of {numeric_cols[0]} over {date_cols[0]} grouped by {categorical_cols[0]}",
                        "columns": [date_cols[0], numeric_cols[0], categorical_cols[0]]
                    })
        else:
            # Get suggestions for specific columns
            suggestions = []

            # Check data types of specified columns
            column_types = {}
            for col in columns:
                if col not in df.columns:
                    continue

                if pd.api.types.is_numeric_dtype(df[col]):
                    column_types[col] = "numeric"
                elif pd.api.types.is_datetime64_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
                    column_types[col] = "date"
                else:
                    column_types[col] = "categorical"

            # Get numeric and categorical columns
            numeric_cols = [col for col, type_ in column_types.items() if type_ == "numeric"]
            categorical_cols = [col for col, type_ in column_types.items() if type_ == "categorical"]
            date_cols = [col for col, type_ in column_types.items() if type_ == "date"]

            # For each numeric column
            for num_col in numeric_cols:
                suggestions.append({
                    "type": "histogram",
                    "description": f"Distribution of {num_col}",
                    "columns": [num_col]
                })

            # For pairs of numeric columns
            for i, num_col1 in enumerate(numeric_cols):
                for num_col2 in numeric_cols[i + 1:]:
                    suggestions.append({
                        "type": "scatter",
                        "description": f"Relationship between {num_col1} and {num_col2}",
                        "columns": [num_col1, num_col2]
                    })

            # For numeric and categorical combinations
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    suggestions.append({
                        "type": "boxplot",
                        "description": f"Distribution of {num_col} by {cat_col}",
                        "columns": [cat_col, num_col]
                    })

                    suggestions.append({
                        "type": "barplot",
                        "description": f"Average {num_col} by {cat_col}",
                        "columns": [cat_col, num_col]
                    })

            # For date and numeric combinations
            for date_col in date_cols:
                for num_col in numeric_cols:
                    suggestions.append({
                        "type": "lineplot",
                        "description": f"Trend of {num_col} over {date_col}",
                        "columns": [date_col, num_col]
                    })

                    for cat_col in categorical_cols:
                        suggestions.append({
                            "type": "lineplot",
                            "description": f"Trend of {num_col} over {date_col} grouped by {cat_col}",
                            "columns": [date_col, num_col, cat_col]
                        })

            # If multiple numeric columns, suggest heatmap
            if len(numeric_cols) >= 2:
                suggestions.append({
                    "type": "heatmap",
                    "description": f"Correlation matrix of {', '.join(numeric_cols)}",
                    "columns": numeric_cols
                })

        # Add operation to memory
        self.memory.add_operation(
            "Get visualization suggestions",
            "suggest_visualizations()",
            f"Generated {len(suggestions)} visualization suggestions"
        )

        return {
            "success": True,
            "message": f"Generated {len(suggestions)} visualization suggestions based on your data",
            "result": {
                "suggestions": suggestions
            }
        }