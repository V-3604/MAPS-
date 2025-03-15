class FunctionRegistry:
    def __init__(self):
        self.pandas_functions = self._initialize_pandas_functions()
        self.viz_functions = self._initialize_viz_functions()

    def _initialize_pandas_functions(self):
        """Initialize dictionary of common pandas operations"""
        return {
            "read_csv": {
                "code": "pd.read_csv('{filename}', {args})",
                "description": "Load data from a CSV file",
                "params": {
                    "filename": "Path to the CSV file",
                    "args": "Additional arguments for pd.read_csv"
                },
                "example": "pd.read_csv('data.csv', index_col=0)"
            },
            "read_excel": {
                "code": "pd.read_excel('{filename}', {args})",
                "description": "Load data from an Excel file",
                "params": {
                    "filename": "Path to the Excel file",
                    "args": "Additional arguments for pd.read_excel"
                },
                "example": "pd.read_excel('data.xlsx', sheet_name='Sheet1')"
            },
            "drop_na": {
                "code": "df.dropna({args})",
                "description": "Remove rows with missing values",
                "params": {
                    "args": "Arguments for dropna like subset=['col1', 'col2']"
                },
                "example": "df.dropna(subset=['important_column'])"
            },
            "fill_na": {
                "code": "df.fillna({value}, {args})",
                "description": "Fill missing values",
                "params": {
                    "value": "Value to fill NAs with",
                    "args": "Additional arguments for fillna"
                },
                "example": "df.fillna(0, inplace=True)"
            },
            "describe": {
                "code": "df.describe({args})",
                "description": "Generate descriptive statistics",
                "params": {
                    "args": "Additional arguments for describe"
                },
                "example": "df.describe(include='all')"
            },
            "group_by": {
                "code": "df.groupby([{columns}]).{agg}()",
                "description": "Group data by columns and apply aggregation",
                "params": {
                    "columns": "Columns to group by",
                    "agg": "Aggregation function to apply"
                },
                "example": "df.groupby(['category', 'region']).mean()"
            },
            "sort_values": {
                "code": "df.sort_values(by=[{columns}], ascending={ascending})",
                "description": "Sort dataframe by values",
                "params": {
                    "columns": "Columns to sort by",
                    "ascending": "Boolean or list of booleans for sort direction"
                },
                "example": "df.sort_values(by=['revenue'], ascending=False)"
            },
            "column_dtypes": {
                "code": "df.dtypes",
                "description": "Get data types of each column",
                "params": {},
                "example": "df.dtypes"
            },
            "create_column": {
                "code": "df['{new_col}'] = {expression}",
                "description": "Create a new column using an expression",
                "params": {
                    "new_col": "New column name",
                    "expression": "Expression to calculate column values"
                },
                "example": "df['total'] = df['price'] * df['quantity']"
            },
            "rename_columns": {
                "code": "df.rename(columns={{{column_mapping}}})",
                "description": "Rename columns",
                "params": {
                    "column_mapping": "Dictionary mapping old names to new names"
                },
                "example": "df.rename(columns={'old_name': 'new_name'})"
            }
        }

    def _initialize_viz_functions(self):
        """Initialize dictionary of common visualization operations"""
        return {
            "histogram": {
                "code": "sns.histplot(data={data}, x='{x}', {args})",
                "description": "Create histogram of a numeric column",
                "params": {
                    "data": "DataFrame to use",
                    "x": "Column to plot",
                    "args": "Additional arguments like kde=True, bins=30"
                },
                "example": "sns.histplot(data=df, x='age', kde=True, bins=20)"
            },
            "scatter": {
                "code": "sns.scatterplot(data={data}, x='{x}', y='{y}', {args})",
                "description": "Create scatter plot between two numeric columns",
                "params": {
                    "data": "DataFrame to use",
                    "x": "Column for x-axis",
                    "y": "Column for y-axis",
                    "args": "Additional arguments like hue='category'"
                },
                "example": "sns.scatterplot(data=df, x='height', y='weight', hue='gender')"
            },
            "line": {
                "code": "sns.lineplot(data={data}, x='{x}', y='{y}', {args})",
                "description": "Create line plot",
                "params": {
                    "data": "DataFrame to use",
                    "x": "Column for x-axis",
                    "y": "Column for y-axis",
                    "args": "Additional arguments"
                },
                "example": "sns.lineplot(data=df, x='year', y='revenue', hue='product')"
            },
            "barplot": {
                "code": "sns.barplot(data={data}, x='{x}', y='{y}', {args})",
                "description": "Create bar plot",
                "params": {
                    "data": "DataFrame to use",
                    "x": "Column for x-axis",
                    "y": "Column for y-axis",
                    "args": "Additional arguments"
                },
                "example": "sns.barplot(data=df, x='category', y='sales')"
            },
            "boxplot": {
                "code": "sns.boxplot(data={data}, x='{x}', y='{y}', {args})",
                "description": "Create box plot",
                "params": {
                    "data": "DataFrame to use",
                    "x": "Column for x-axis (usually categorical)",
                    "y": "Column for y-axis (usually numeric)",
                    "args": "Additional arguments"
                },
                "example": "sns.boxplot(data=df, x='category', y='distribution')"
            },
            "heatmap": {
                "code": "sns.heatmap({data}, {args})",
                "description": "Create heatmap (often used for correlation matrices)",
                "params": {
                    "data": "Matrix to display (e.g., df.corr())",
                    "args": "Additional arguments like annot=True, cmap='coolwarm'"
                },
                "example": "sns.heatmap(df.corr(), annot=True, cmap='coolwarm')"
            },
            "pairplot": {
                "code": "sns.pairplot(data={data}, {args})",
                "description": "Create pairwise relationships plot",
                "params": {
                    "data": "DataFrame to use",
                    "args": "Additional arguments like hue='category', vars=['col1', 'col2']"
                },
                "example": "sns.pairplot(df, hue='species')"
            },
            "countplot": {
                "code": "sns.countplot(data={data}, x='{x}', {args})",
                "description": "Create count plot for categorical data",
                "params": {
                    "data": "DataFrame to use",
                    "x": "Column to count",
                    "args": "Additional arguments"
                },
                "example": "sns.countplot(data=df, x='category', order=df['category'].value_counts().index)"
            }
        }

    def get_pandas_function(self, func_name):
        """Get a pandas function by name"""
        return self.pandas_functions.get(func_name)

    def get_viz_function(self, func_name):
        """Get a visualization function by name"""
        return self.viz_functions.get(func_name)

    def list_pandas_functions(self):
        """List all available pandas functions"""
        return [(name, info["description"]) for name, info in self.pandas_functions.items()]

    def list_viz_functions(self):
        """List all available visualization functions"""
        return [(name, info["description"]) for name, info in self.viz_functions.items()]

    def suggest_viz_for_columns(self, df, columns):
        """Suggest appropriate visualizations for given columns"""
        suggestions = []

        # Convert columns to list if it's a string
        if isinstance(columns, str):
            columns = [columns]

        # Check if all columns exist in dataframe
        for col in columns:
            if col not in df.columns:
                return [f"Column '{col}' not found in dataframe"]

        # Single column analysis
        if len(columns) == 1:
            col = columns[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                suggestions.append(("histogram", f"Distribution of {col}"))
                suggestions.append(("boxplot", f"Box plot of {col}"))
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
                suggestions.append(("countplot", f"Count of {col} categories"))
                suggestions.append(("barplot", f"Bar plot of {col} distribution"))

        # Two column analysis
        elif len(columns) == 2:
            col1, col2 = columns
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                suggestions.append(("scatter", f"Relationship between {col1} and {col2}"))
                suggestions.append(("heatmap", f"Correlation heatmap including {col1} and {col2}"))
            elif pd.api.types.is_numeric_dtype(df[col1]) and (
                    pd.api.types.is_categorical_dtype(df[col2]) or df[col2].nunique() < 20):
                suggestions.append(("boxplot", f"Distribution of {col1} by {col2} categories"))
                suggestions.append(("barplot", f"Average {col1} by {col2} categories"))
            elif pd.api.types.is_numeric_dtype(df[col2]) and (
                    pd.api.types.is_categorical_dtype(df[col1]) or df[col1].nunique() < 20):
                suggestions.append(("boxplot", f"Distribution of {col2} by {col1} categories"))
                suggestions.append(("barplot", f"Average {col2} by {col1} categories"))

        # Multiple columns
        else:
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            categorical_cols = [col for col in columns if
                                pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20]

            if len(numeric_cols) >= 2:
                suggestions.append(("pairplot", f"Pairwise relationships among {', '.join(numeric_cols)}"))
                suggestions.append(("heatmap", f"Correlation matrix of {', '.join(numeric_cols)}"))

            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                suggestions.append(("boxplot", f"Multiple box plots by categories"))

        return suggestions