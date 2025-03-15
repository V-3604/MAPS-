import json


class OrchestratorAgent:
    def __init__(self, data_engineer, viz_specialist, memory_agent, conversation_manager):
        self.data_engineer = data_engineer
        self.viz_specialist = viz_specialist
        self.memory_agent = memory_agent
        self.conversation = conversation_manager

    def process_query(self, query):
        """Process a user query and route to appropriate agent"""
        # Add user query to conversation
        self.conversation.add_message("user", query)

        # Classify query type
        query_type = self._classify_query(query)
        print(f"Query classified as: {query_type}")  # Debug info

        # Route to appropriate agent
        if query_type == "data_processing":
            response = self.data_engineer.process_query(query, self.conversation)

            # Update memory with operation details
            if response["success"]:
                self.memory_agent.update_after_operation(
                    "data_processing",
                    query,
                    response["result"]
                )

        elif query_type == "visualization":
            # Get the current dataframe from data engineer
            df = self.data_engineer.current_df

            if df is None:
                response = {
                    "success": False,
                    "message": "No data available for visualization. Please load or process data first.",
                    "result": None
                }
            else:
                response = self.viz_specialist.process_query(query, df, self.conversation)

                # Update memory with operation details
                if response["success"]:
                    self.memory_agent.update_after_operation(
                        "visualization",
                        query,
                        response["result"]
                    )

        elif query_type == "memory":
            response = self.memory_agent.process_query(query)

        else:  # general or ambiguous
            # Try data processing first as a fallback
            data_response = self.data_engineer.process_query(query, self.conversation)
            if data_response["success"]:
                response = data_response
                self.memory_agent.update_after_operation(
                    "data_processing",
                    query,
                    response["result"]
                )
            else:
                # Try to provide a helpful response
                response = self._handle_general_query(query)

        # Add response to conversation
        self.conversation.add_message(
            "system",
            response["message"],
            {"type": query_type, "success": response["success"],
             "result_summary": str(response["result"])[:100] if response["result"] else None}
        )

        return response

    def _classify_query(self, query):
        """Classify the query type to route to appropriate agent"""
        query_lower = query.lower()

        # Data processing indicators
        data_keywords = [
            "load", "import", "read", "data", "csv", "excel", "clean", "process",
            "transform", "filter", "missing", "values", "drop", "columns", "rows",
            "sort", "group", "merge", "join", "dataframe", "calculate", "create column",
            "sample", "head", "show me the", "first", "check for", "explore", "describe",
            "summary", "create a new column", "fill", "replace", "rename"
        ]

        # Visualization indicators
        viz_keywords = [
            "plot", "chart", "graph", "visualize", "draw", "histogram", "bar", "scatter",
            "box", "distribution", "trend", "line", "heatmap", "correlation", "show me",
            "display", "visualisation", "visualization", "average", "relationship"
        ]

        # Memory indicators
        memory_keywords = [
            "memory", "context", "history", "summary", "status", "operations",
            "what did you do", "what have you done", "previous", "remember",
            "recall", "steps", "log", "checkpoint", "save state", "summarize",
            "so far", "save", "load checkpoint"
        ]

        # Count keyword matches
        data_matches = [word for word in data_keywords if word in query_lower]
        viz_matches = [word for word in viz_keywords if word in query_lower]
        memory_matches = [word for word in memory_keywords if word in query_lower]

        data_count = len(data_matches)
        viz_count = len(viz_matches)
        memory_count = len(memory_matches)

        # Special case for showing data
        if "show" in query_lower and any(col in query_lower for col in ["rows", "data", "values", "head"]):
            return "data_processing"

        # Special case for creating columns
        if "create" in query_lower and "column" in query_lower:
            return "data_processing"

        # Special case for checking missing values
        if ("check" in query_lower or "missing" in query_lower) and "values" in query_lower:
            return "data_processing"

        # Special case for memory queries
        if "what we've done" in query_lower or "summarize what" in query_lower or "checkpoint" in query_lower:
            return "memory"

        # Special case for visualizations with columns
        if any(viz in query_lower for viz in ["histogram", "scatter plot", "bar chart", "boxplot"]):
            return "visualization"

        # Determine category based on keyword matches
        if data_count > viz_count and data_count > memory_count:
            return "data_processing"
        elif viz_count > data_count and viz_count > memory_count:
            return "visualization"
        elif memory_count > data_count and memory_count > viz_count:
            return "memory"
        else:
            # Default to data processing for ambiguous queries
            return "data_processing"

    def _handle_general_query(self, query):
        """Handle general or ambiguous queries"""
        query_lower = query.lower()

        # Check for help or guidance requests
        if "help" in query_lower or "guide" in query_lower or "what can you do" in query_lower:
            return self._provide_help()

        # Check for state inquiries
        if "what data" in query_lower or "current data" in query_lower:
            # Get info about current data
            if self.data_engineer.current_df is not None:
                df = self.data_engineer.current_df
                info = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "sample": df.head(3).to_dict(orient="records")
                }
                return {
                    "success": True,
                    "message": f"Current dataframe has shape {df.shape} with {len(df.columns)} columns",
                    "result": info
                }
            else:
                return {
                    "success": False,
                    "message": "No data has been loaded yet",
                    "result": None
                }

        # Fall back to asking clarification
        return {
            "success": False,
            "message": "I'm not sure what you're asking for. Could you clarify if you want to process data, create a visualization, or check memory/context?",
            "result": None
        }

    def _provide_help(self):
        """Provide help information about system capabilities"""
        help_info = {
            "data_processing": {
                "description": "Load, clean, and transform data",
                "examples": [
                    "Load data from 'sales_data.csv'",
                    "Show me the first 5 rows of data",
                    "Check for missing values",
                    "Drop rows with missing values",
                    "Create a new column 'total' as price times quantity"
                ]
            },
            "visualization": {
                "description": "Create visualizations from data",
                "examples": [
                    "Show a histogram of the 'age' column",
                    "Create a scatter plot of 'height' vs 'weight'",
                    "Generate a bar chart of sales by region",
                    "Show a correlation heatmap",
                    "Suggest visualizations for my data"
                ]
            },
            "memory": {
                "description": "Check system context and history",
                "examples": [
                    "Summarize the current context",
                    "Show me the operation history",
                    "List all visualizations we've created",
                    "Save a checkpoint",
                    "What have we done so far?"
                ]
            }
        }

        return {
            "success": True,
            "message": "Here's what I can help you with",
            "result": help_info
        }