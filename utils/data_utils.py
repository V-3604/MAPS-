import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


def load_data(file_path, **kwargs):
    """
    Load data from various file formats.

    Parameters:
    -----------
    file_path : str
        Path to the data file
    **kwargs : dict
        Additional arguments to pass to the appropriate pandas read function

    Returns:
    --------
    pd.DataFrame
        The loaded dataframe
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path, **kwargs)
    elif file_extension == '.json':
        return pd.read_json(file_path, **kwargs)
    elif file_extension == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif file_extension == '.pkl':
        return pd.read_pickle(file_path, **kwargs)
    elif file_extension == '.feather':
        return pd.read_feather(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def save_data(df, file_path, **kwargs):
    """
    Save a dataframe to a file.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to save
    file_path : str
        Path where the file should be saved
    **kwargs : dict
        Additional arguments to pass to the appropriate pandas write function

    Returns:
    --------
    str
        The file path where data was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        df.to_csv(file_path, **kwargs)
    elif file_extension in ['.xls', '.xlsx']:
        df.to_excel(file_path, **kwargs)
    elif file_extension == '.json':
        df.to_json(file_path, **kwargs)
    elif file_extension == '.parquet':
        df.to_parquet(file_path, **kwargs)
    elif file_extension == '.pkl':
        df.to_pickle(file_path, **kwargs)
    elif file_extension == '.feather':
        df.to_feather(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return file_path


def generate_data_summary(df):
    """
    Generate a comprehensive summary of a dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to summarize

    Returns:
    --------
    dict
        A dictionary containing summary information
    """
    if df is None:
        return {"error": "No dataframe provided"}

    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "missing_percentage": {col: float(df[col].isna().mean() * 100) for col in df.columns},
        "unique_counts": {col: int(df[col].nunique()) for col in df.columns}
    }

    # Add descriptive statistics for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

    # Add categorical value counts for categorical columns (limiting to top 5)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        summary["categorical_stats"] = {}
        for col in cat_cols:
            value_counts = df[col].value_counts().head(5).to_dict()
            summary["categorical_stats"][col] = value_counts

    return summary


def detect_data_issues(df):
    """
    Detect common data quality issues in a dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to check

    Returns:
    --------
    dict
        A dictionary containing detected issues
    """
    issues = {
        "missing_values": {},
        "outliers": {},
        "inconsistent_values": {},
        "duplicates": 0
    }

    # Check for missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            issues["missing_values"][col] = {
                "count": int(missing),
                "percentage": float(missing / len(df) * 100)
            }

    # Check for outliers in numeric columns
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outliers_count > 0:
            issues["outliers"][col] = {
                "count": int(outliers_count),
                "percentage": float(outliers_count / len(df) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }

    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues["duplicates"] = int(duplicate_count)

    # Check for inconsistent values in categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        # Look for similar values that might be inconsistent (very basic check)
        values = df[col].dropna().unique()

        # Check for values that differ only by case
        lowercase_values = [str(v).lower() for v in values if isinstance(v, str)]
        if len(set(lowercase_values)) < len(lowercase_values):
            issues["inconsistent_values"][col] = "Possible case inconsistencies"

    return issues


def recommend_transformations(df):
    """
    Recommend data transformations based on the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze

    Returns:
    --------
    list
        A list of recommended transformations
    """
    recommendations = []

    # Check for missing values and recommend handling
    missing_cols = [col for col in df.columns if df[col].isna().sum() > 0]
    if missing_cols:
        recommendations.append({
            "type": "missing_values",
            "description": f"Handle missing values in {len(missing_cols)} columns",
            "details": {col: int(df[col].isna().sum()) for col in missing_cols},
            "suggested_code": f"df.dropna(subset={missing_cols})"
        })

    # Check for datetime columns that need conversion
    date_pattern_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        # Check sample values for date-like patterns
        sample = df[col].dropna().head(10).astype(str)
        if any('/' in val or '-' in val or ':' in val for val in sample):
            try:
                pd.to_datetime(sample, errors='raise')
                date_pattern_cols.append(col)
            except:
                pass

    if date_pattern_cols:
        recommendations.append({
            "type": "datetime_conversion",
            "description": f"Convert possible date columns to datetime: {', '.join(date_pattern_cols)}",
            "details": {"columns": date_pattern_cols},
            "suggested_code": f"df['{date_pattern_cols[0]}'] = pd.to_datetime(df['{date_pattern_cols[0]}'])"
        })

    # Check for skewed numeric columns that might need transformation
    skewed_cols = []
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].skew() > 1.0:
            skewed_cols.append(col)

    if skewed_cols:
        recommendations.append({
            "type": "skew_transformation",
            "description": f"Apply log transformation to skewed numeric columns: {', '.join(skewed_cols)}",
            "details": {"columns": skewed_cols},
            "suggested_code": f"df['{skewed_cols[0]}_log'] = np.log1p(df['{skewed_cols[0]}'])"
        })

    # Check for high cardinality categorical columns that might need grouping
    high_card_cols = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() > 10 and df[col].nunique() < len(df) * 0.5:
            high_card_cols.append((col, df[col].nunique()))

    if high_card_cols:
        col, count = high_card_cols[0]
        recommendations.append({
            "type": "category_grouping",
            "description": f"Group categories in high cardinality columns: {', '.join([c[0] for c in high_card_cols])}",
            "details": {"columns": dict(high_card_cols)},
            "suggested_code": f"# Group all but top 5 categories in '{col}'\n" +
                              f"top_categories = df['{col}'].value_counts().nlargest(5).index\n" +
                              f"df['{col}_grouped'] = df['{col}'].apply(lambda x: x if x in top_categories else 'Other')"
        })

    return recommendations