import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime


def setup_visualization_defaults():
    """Set up default visualization settings"""
    # Set Seaborn style
    sns.set_theme(style="whitegrid", context="notebook")

    # Set figure defaults
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100

    # Set font defaults
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    # Other defaults
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def generate_filename(viz_type):
    """Generate a unique filename for a visualization"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{viz_type}_{timestamp}.png"


def save_visualization(plt, output_dir, filename=None, viz_type="plot"):
    """Save a matplotlib visualization to a file"""
    if filename is None:
        filename = generate_filename(viz_type)

    filepath = os.path.join(output_dir, filename)

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    return filepath


def suggest_visualizations(df, columns=None):
    """Suggest appropriate visualizations based on data types"""
    if columns is None:
        columns = df.columns.tolist()

    suggestions = []

    # Get column types
    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in columns if col in df.columns and (pd.api.types.is_categorical_dtype(df[col]) or
                                                                         df[col].dtype == 'object')]
    date_cols = [col for col in columns if col in df.columns and pd.api.types.is_datetime64_dtype(df[col])]

    # Also detect potential date columns that are stored as strings
    for col in columns:
        if col in df.columns and col not in date_cols and df[col].dtype == 'object':
            if 'date' in col.lower() or 'time' in col.lower():
                date_cols.append(col)

    # Add suggestions for numeric columns
    for col in numeric_cols:
        # Histogram for distribution
        suggestions.append({
            "type": "histogram",
            "description": f"Distribution of {col}",
            "columns": [col],
            "code": f"sns.histplot(data=df, x='{col}', kde=True)"
        })

    # Add suggestions for categorical columns
    for col in categorical_cols:
        # Count plot
        suggestions.append({
            "type": "countplot",
            "description": f"Count of records by {col}",
            "columns": [col],
            "code": f"sns.countplot(data=df, x='{col}', order=df['{col}'].value_counts().index[:20])"
        })

    # Add suggestions for numeric + categorical combinations
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            # Box plot
            suggestions.append({
                "type": "boxplot",
                "description": f"Distribution of {num_col} by {cat_col}",
                "columns": [cat_col, num_col],
                "code": f"sns.boxplot(data=df, x='{cat_col}', y='{num_col}')"
            })

            # Bar plot
            suggestions.append({
                "type": "barplot",
                "description": f"Average {num_col} by {cat_col}",
                "columns": [cat_col, num_col],
                "code": f"sns.barplot(data=df, x='{cat_col}', y='{num_col}')"
            })

    # Add suggestions for numeric + numeric combinations
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                # Scatter plot
                suggestions.append({
                    "type": "scatter",
                    "description": f"Relationship between {col1} and {col2}",
                    "columns": [col1, col2],
                    "code": f"sns.scatterplot(data=df, x='{col1}', y='{col2}')"
                })

        # Correlation heatmap for all numeric columns
        if len(numeric_cols) >= 3:
            suggestions.append({
                "type": "heatmap",
                "description": "Correlation matrix of numeric columns",
                "columns": numeric_cols,
                "code": "sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')"
            })

    # Add suggestions for time series
    for date_col in date_cols:
        for num_col in numeric_cols:
            # Line plot
            suggestions.append({
                "type": "lineplot",
                "description": f"Trend of {num_col} over {date_col}",
                "columns": [date_col, num_col],
                "code": f"sns.lineplot(data=df, x='{date_col}', y='{num_col}')"
            })

            # For each categorical column, suggest a faceted time series
            for cat_col in categorical_cols:
                suggestions.append({
                    "type": "lineplot_faceted",
                    "description": f"Trend of {num_col} over {date_col} by {cat_col}",
                    "columns": [date_col, num_col, cat_col],
                    "code": f"sns.relplot(data=df, x='{date_col}', y='{num_col}', hue='{cat_col}', kind='line')"
                })

    return suggestions


def create_visualization(viz_type, df, **kwargs):
    """Create a visualization based on type and parameters"""
    plt.figure(figsize=(10, 6))

    if viz_type == "histogram":
        col = kwargs.get("column")
        bins = kwargs.get("bins", 30)
        kde = kwargs.get("kde", True)

        ax = sns.histplot(data=df, x=col, bins=bins, kde=kde)
        plt.title(f"Distribution of {col}")

    elif viz_type == "scatter":
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        hue_col = kwargs.get("hue_column")

        ax = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.title(f"Relationship between {x_col} and {y_col}" +
                  (f" by {hue_col}" if hue_col else ""))

    elif viz_type == "barplot":
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        hue_col = kwargs.get("hue_column")

        ax = sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.title(f"Average {y_col} by {x_col}" +
                  (f" and {hue_col}" if hue_col else ""))

        # Rotate x-axis labels if there are many categories
        if df[x_col].nunique() > 6:
            plt.xticks(rotation=45, ha="right")

    elif viz_type == "boxplot":
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        hue_col = kwargs.get("hue_column")

        ax = sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.title(f"Distribution of {y_col} by {x_col}" +
                  (f" and {hue_col}" if hue_col else ""))

        # Rotate x-axis labels if there are many categories
        if x_col and df[x_col].nunique() > 6:
            plt.xticks(rotation=45, ha="right")

    elif viz_type == "lineplot":
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        hue_col = kwargs.get("hue_column")

        ax = sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.title(f"Trend of {y_col} over {x_col}" +
                  (f" by {hue_col}" if hue_col else ""))

        # Rotate x-axis labels for time series
        plt.xticks(rotation=45, ha="right")

    elif viz_type == "heatmap":
        columns = kwargs.get("columns")
        df_corr = df[columns].corr() if columns else df.corr()

        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        ax = sns.heatmap(df_corr, mask=mask, annot=True, cmap="coolwarm",
                         vmin=-1, vmax=1, linewidths=0.5)
        plt.title("Correlation Matrix")

    elif viz_type == "countplot":
        col = kwargs.get("column")
        hue_col = kwargs.get("hue_column")

        # Order by frequency
        order = df[col].value_counts().head(20).index

        ax = sns.countplot(data=df, x=col, hue=hue_col, order=order)
        plt.title(f"Count of records by {col}" +
                  (f" and {hue_col}" if hue_col else ""))

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

    elif viz_type == "pairplot":
        columns = kwargs.get("columns")
        hue_col = kwargs.get("hue_column")

        # Close the current figure as pairplot creates its own
        plt.close()

        g = sns.pairplot(df[columns + [hue_col]] if hue_col else df[columns],
                         hue=hue_col)
        return g

    else:
        plt.close()
        raise ValueError(f"Unsupported visualization type: {viz_type}")

    plt.tight_layout()
    return plt