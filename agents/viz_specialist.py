# agents/viz_specialist.py
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64


class VizSpecialist:
    def __init__(self, memory_system, function_registry, **kwargs):
        self.memory_system = memory_system
        self.function_registry = function_registry
        self.current_visualization = None
        self.visualization_history = []
        self.output_dir = Path("output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.config = {
            "default_width": 10,
            "default_height": 6,
            "default_dpi": 100,
            "theme": "whitegrid",
            "show_grid": True,
            "save_format": "png",
            "include_timestamp": True
        }
        # Update config with any provided kwargs
        self.config.update(kwargs)

        # Set the default style
        sns.set_theme(style=self.config["theme"])

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        Path('output/logs').mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler('output/logs/viz_specialist.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def create_visualization(self, viz_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a visualization based on the specified type and parameters

        Parameters:
        -----------
        viz_type : str
            Type of visualization (e.g., "line", "scatter", "bar", "histogram")
        params : Dict[str, Any]
            Parameters for the visualization

        Returns:
        --------
        Dict[str, Any]
            Result of the visualization operation
        """
        try:
            # Check if data is already provided in params
            if "data" not in params:
                # Get the current dataset from memory using the correct method
                data_result = self.memory_system.handle_memory_request({
                    "operation": "retrieve",
                    "params": {"key": "test_data"}
                })

                if not data_result.get("success", False):
                    self.logger.error(f"Failed to retrieve dataset: {data_result.get('error', 'Unknown error')}")
                    return {"success": False, "error": "Failed to retrieve dataset"}

                data = data_result.get("data")
                if data is None:
                    return {"success": False, "error": "No data set for visualization"}

                params["data"] = data

            # Call the appropriate visualization method
            viz_methods = {
                "line": self._create_line_plot,
                "scatter": self._create_scatter_plot,
                "bar": self._create_bar_plot,
                "histogram": self._create_histogram,
                "heatmap": self._create_heatmap,
                "boxplot": self._create_boxplot,
                "pairplot": self._create_pairplot,
                "pie": self._create_pie_chart
            }

            if viz_type not in viz_methods:
                # Try to use the function registry
                viz_function = self.function_registry.get_function(f"viz_{viz_type}", "visualization")
                if viz_function:
                    return viz_function(**params)
                else:
                    return {"success": False, "error": f"Unsupported visualization type: {viz_type}"}

            # Create the visualization
            viz_result = viz_methods[viz_type](params.get("data"), params)

            if not viz_result.get("success", False):
                return viz_result

            # Store in history
            self.visualization_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": viz_type,
                "params": params,
                "path": viz_result.get("path")
            })

            # Save current visualization reference
            self.current_visualization = {
                "type": viz_type,
                "params": params,
                "path": viz_result.get("path")
            }

            # Don't include the full image data in the result if not needed
            if 'image_data' in viz_result and len(viz_result['image_data']) > 100:
                viz_result['image_data'] = viz_result['image_data'][:50] + '...' + viz_result['image_data'][-50:]

            return viz_result

        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_line_plot(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a line plot"""
        try:
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Line Plot")
            xlabel = params.get("xlabel", x)
            ylabel = params.get("ylabel", y)
            color = params.get("color")
            figsize = params.get("figsize", (self.config["default_width"], self.config["default_height"]))

            if not x or not y:
                return {"success": False, "error": "Missing required parameters: x and y"}

            if x not in data.columns or y not in data.columns:
                return {"success": False,
                        "error": f"Columns not found in dataset: {x if x not in data.columns else ''} {y if y not in data.columns else ''}".strip()}

            plt.figure(figsize=figsize)

            if color and color in data.columns:
                for category in data[color].unique():
                    subset = data[data[color] == category]
                    plt.plot(subset[x], subset[y], label=str(category))
                plt.legend(title=color)
            else:
                plt.plot(data[x], data[y])

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(self.config["show_grid"])

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"line_plot_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            plt.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(plt.gcf())

            plt.close()

            return {
                "success": True,
                "message": "Line plot created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating line plot: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_scatter_plot(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a scatter plot"""
        try:
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Scatter Plot")
            xlabel = params.get("xlabel", x)
            ylabel = params.get("ylabel", y)
            color = params.get("color")
            size = params.get("size")
            alpha = params.get("alpha", 0.7)
            figsize = params.get("figsize", (self.config["default_width"], self.config["default_height"]))

            if not x or not y:
                return {"success": False, "error": "Missing required parameters: x and y"}

            if x not in data.columns or y not in data.columns:
                return {"success": False,
                        "error": f"Columns not found in dataset: {x if x not in data.columns else ''} {y if y not in data.columns else ''}".strip()}

            plt.figure(figsize=figsize)

            scatter_kwargs = {"alpha": alpha}

            if color and color in data.columns:
                # Use a color map instead of directly using the column values
                categories = data[color].unique()
                cmap = plt.cm.get_cmap('tab10', len(categories))
                categorical_map = {cat: i for i, cat in enumerate(categories)}

                scatter = plt.scatter(
                    data[x],
                    data[y],
                    c=[categorical_map[cat] for cat in data[color]],
                    cmap=cmap,
                    **scatter_kwargs
                )
                plt.colorbar(scatter, label=color, ticks=range(len(categories)))
                plt.clim(-0.5, len(categories) - 0.5)

                # Add color legend
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=cmap(categorical_map[cat] / len(categories)),
                                      markersize=10, label=cat) for cat in categories]
                plt.legend(handles=handles, title=color)

            elif size and size in data.columns:
                scatter_kwargs["s"] = data[size]
                plt.scatter(data[x], data[y], **scatter_kwargs)
            else:
                plt.scatter(data[x], data[y], **scatter_kwargs)

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(self.config["show_grid"])

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scatter_plot_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            plt.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(plt.gcf())

            plt.close()

            return {
                "success": True,
                "message": "Scatter plot created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating scatter plot: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_bar_plot(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a bar plot"""
        try:
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Bar Plot")
            xlabel = params.get("xlabel", x)
            ylabel = params.get("ylabel", y)
            color = params.get("color")
            orientation = params.get("orientation", "vertical")
            figsize = params.get("figsize", (self.config["default_width"], self.config["default_height"]))

            if not x or not y:
                return {"success": False, "error": "Missing required parameters: x and y"}

            if x not in data.columns or y not in data.columns:
                return {"success": False,
                        "error": f"Columns not found in dataset: {x if x not in data.columns else ''} {y if y not in data.columns else ''}".strip()}

            plt.figure(figsize=figsize)

            if color and color in data.columns:
                grouped = data.groupby([x, color])[y].sum().unstack()
                grouped.plot(kind="bar", ax=plt.gca())
            else:
                if orientation == "horizontal":
                    plt.barh(data[x], data[y])
                else:
                    plt.bar(data[x], data[y])

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(self.config["show_grid"])

            # Rotate x labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bar_plot_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            plt.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(plt.gcf())

            plt.close()

            return {
                "success": True,
                "message": "Bar plot created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating bar plot: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_histogram(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a histogram"""
        try:
            x = params.get("x")
            bins = params.get("bins", 10)
            title = params.get("title", "Histogram")
            xlabel = params.get("xlabel", x)
            ylabel = params.get("ylabel", "Frequency")
            color = params.get("color")
            kde = params.get("kde", False)
            figsize = params.get("figsize", (self.config["default_width"], self.config["default_height"]))

            if not x:
                return {"success": False, "error": "Missing required parameter: x"}

            if x not in data.columns:
                return {"success": False, "error": f"Column not found in dataset: {x}"}

            plt.figure(figsize=figsize)

            if color and color in data.columns:
                for category in data[color].unique():
                    subset = data[data[color] == category]
                    sns.histplot(subset[x], bins=bins, kde=kde, label=str(category))
                plt.legend(title=color)
            else:
                sns.histplot(data[x], bins=bins, kde=kde)

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(self.config["show_grid"])

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"histogram_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            plt.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(plt.gcf())

            plt.close()

            return {
                "success": True,
                "message": "Histogram created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating histogram: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_heatmap(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a heatmap"""
        try:
            columns = params.get("columns", None)
            title = params.get("title", "Correlation Heatmap")
            figsize = params.get("figsize", (self.config["default_width"], self.config["default_height"]))
            cmap = params.get("cmap", "viridis")
            annot = params.get("annot", True)

            # Filter numeric columns only
            numeric_data = data.select_dtypes(include=['number'])

            if numeric_data.empty:
                return {"success": False, "error": "No numeric columns available for heatmap"}

            if columns:
                # Filter specified columns
                valid_columns = [col for col in columns if col in numeric_data.columns]
                if not valid_columns:
                    return {"success": False, "error": "None of the specified columns are numeric or available"}
                numeric_data = numeric_data[valid_columns]

            corr = numeric_data.corr()

            plt.figure(figsize=figsize)
            sns.heatmap(corr, annot=annot, cmap=cmap)
            plt.title(title)
            plt.tight_layout()

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmap_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            plt.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(plt.gcf())

            plt.close()

            return {
                "success": True,
                "message": "Heatmap created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating heatmap: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_boxplot(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a boxplot"""
        try:
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Box Plot")
            xlabel = params.get("xlabel", x)
            ylabel = params.get("ylabel", y)
            hue = params.get("hue")
            figsize = params.get("figsize", (self.config["default_width"], self.config["default_height"]))

            if not y:
                return {"success": False, "error": "Missing required parameter: y"}

            if y not in data.columns:
                return {"success": False, "error": f"Column not found in dataset: {y}"}

            if x and x not in data.columns:
                return {"success": False, "error": f"Column not found in dataset: {x}"}

            if hue and hue not in data.columns:
                return {"success": False, "error": f"Hue column not found in dataset: {hue}"}

            plt.figure(figsize=figsize)

            sns.boxplot(x=x, y=y, hue=hue, data=data)

            plt.title(title)
            if x:
                plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(self.config["show_grid"])

            # Adjust layout if needed
            plt.tight_layout()

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"boxplot_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            plt.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(plt.gcf())

            plt.close()

            return {
                "success": True,
                "message": "Box plot created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating box plot: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_pairplot(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a pair plot"""
        try:
            columns = params.get("columns", None)
            hue = params.get("hue")
            title = params.get("title", "Pair Plot")
            diag_kind = params.get("diag_kind", "kde")

            # Filter numeric columns only
            numeric_data = data.select_dtypes(include=['number'])

            if numeric_data.empty:
                return {"success": False, "error": "No numeric columns available for pair plot"}

            if columns:
                # Filter specified columns
                valid_columns = [col for col in columns if col in data.columns]
                if not valid_columns:
                    return {"success": False, "error": "None of the specified columns are available"}
                plot_data = data[valid_columns]
            else:
                # Limit to 5 columns max to avoid excessive plotting
                plot_data = numeric_data.iloc[:, :5]

            # Add hue if specified
            if hue:
                if hue in data.columns:
                    plot_data = pd.concat([plot_data, data[hue]], axis=1)
                else:
                    return {"success": False, "error": f"Hue column not found in dataset: {hue}"}

            # Create the pair plot
            pair_grid = sns.pairplot(plot_data, hue=hue, diag_kind=diag_kind)
            pair_grid.fig.suptitle(title, y=1.02)

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pairplot_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            pair_grid.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(pair_grid.fig)

            plt.close()

            return {
                "success": True,
                "message": "Pair plot created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating pair plot: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_pie_chart(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a pie chart"""
        try:
            column = params.get("column")
            title = params.get("title", "Pie Chart")
            figsize = params.get("figsize", (self.config["default_width"], self.config["default_height"]))

            if not column:
                return {"success": False, "error": "Missing required parameter: column"}

            if column not in data.columns:
                return {"success": False, "error": f"Column not found in dataset: {column}"}

            plt.figure(figsize=figsize)

            # Aggregate data if needed
            value_counts = data[column].value_counts()

            # Limit to top 10 categories if there are too many
            if len(value_counts) > 10:
                top_counts = value_counts.nlargest(9)
                other_count = value_counts[9:].sum()
                value_counts = pd.concat([top_counts, pd.Series({"Other": other_count})])

            # Create the pie chart
            plt.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%")
            plt.title(title)

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pie_chart_{timestamp}.{self.config['save_format']}"
            path = self.output_dir / filename
            plt.savefig(path, dpi=self.config["default_dpi"])

            # Convert to base64 for embedding if needed
            img_data = self._fig_to_base64(plt.gcf())

            plt.close()

            return {
                "success": True,
                "message": "Pie chart created successfully",
                "path": str(path),
                "image_data": img_data
            }

        except Exception as e:
            plt.close()
            self.logger.error(f"Error creating pie chart: {str(e)}")
            return {"success": False, "error": str(e)}

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format=self.config["save_format"])
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return img_str

    def get_visualization_history(self) -> Dict[str, Any]:
        """Get visualization history"""
        return {
            "success": True,
            "history": self.visualization_history
        }

    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure visualization settings"""
        try:
            # Update configuration
            self.config.update(config)

            # Update seaborn theme if specified
            if "theme" in config:
                sns.set_theme(style=config["theme"])

            return {
                "success": True,
                "message": "Visualization settings updated successfully",
                "config": self.config
            }
        except Exception as e:
            self.logger.error(f"Error configuring visualization settings: {str(e)}")
            return {"success": False, "error": str(e)}