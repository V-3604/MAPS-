import json
import os
import pandas as pd
from datetime import datetime
import pickle


class MemorySystem:
    def __init__(self, persistence_path="./data/memory/"):
        self.memory = {
            "data_state": {
                "current_file": None,
                "shape": None,
                "dtypes": {},
                "missing_values": {}
            },
            "operation_history": [],
            "key_variables": {},
            "visualizations": []
        }
        self.persistence_path = persistence_path

        # Create persistence directory if it doesn't exist
        os.makedirs(persistence_path, exist_ok=True)

    def update_data_state(self, file_path=None, df=None):
        """Update the current state of the data"""
        if df is not None:
            self.memory["data_state"]["shape"] = df.shape
            self.memory["data_state"]["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            self.memory["data_state"]["missing_values"] = {col: int(df[col].isna().sum()) for col in df.columns}

        if file_path is not None:
            self.memory["data_state"]["current_file"] = file_path

        return self.memory["data_state"]

    def add_operation(self, operation_desc, code, result=None):
        """Add a new operation to history"""
        operation = {
            "step": len(self.memory["operation_history"]) + 1,
            "description": operation_desc,
            "code": code,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if result is not None:
            operation["result_summary"] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)

        self.memory["operation_history"].append(operation)
        return operation

    def add_key_variable(self, var_name, description):
        """Add or update a key variable description"""
        self.memory["key_variables"][var_name] = description
        return self.memory["key_variables"]

    def add_visualization(self, viz_type, description, columns=None, file_path=None):
        """Add a visualization to memory"""
        viz_id = f"viz_{len(self.memory['visualizations']) + 1}"
        viz = {
            "id": viz_id,
            "type": viz_type,
            "description": description,
            "columns": columns,
            "file_path": file_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.memory["visualizations"].append(viz)
        return viz

    def summarize_current_context(self, max_ops=5):
        """Generate a summary of current context for agents"""
        summary = {
            "current_data": self.memory["data_state"],
            "recent_operations": self.memory["operation_history"][-max_ops:] if self.memory[
                "operation_history"] else [],
            "key_variables": self.memory["key_variables"],
            "visualization_count": len(self.memory["visualizations"])
        }
        return summary

    def save_checkpoint(self, checkpoint_name=None):
        """Save current memory state to disk"""
        if checkpoint_name is None:
            checkpoint_name = f"memory_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        file_path = os.path.join(self.persistence_path, f"{checkpoint_name}.json")

        with open(file_path, 'w') as f:
            json.dump(self.memory, f, indent=2)

        return file_path

    def load_checkpoint(self, checkpoint_path):
        """Load memory state from disk"""
        with open(checkpoint_path, 'r') as f:
            self.memory = json.load(f)

        return self.memory

    def save_dataframe(self, df, file_name=None):
        """Save a dataframe checkpoint"""
        if file_name is None:
            file_name = f"df_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        file_path = os.path.join(self.persistence_path, file_name)

        with open(file_path, 'wb') as f:
            pickle.dump(df, f)

        return file_path

    def load_dataframe(self, file_path):
        """Load a dataframe checkpoint"""
        with open(file_path, 'rb') as f:
            df = pickle.load(f)

        return df