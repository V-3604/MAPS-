import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry
from core.conversation_manager import ConversationManager

from agents.data_engineer import DataEngineerAgent
from agents.viz_specialist import VizSpecialistAgent
from agents.memory_agent import MemoryAgent
from agents.orchestrator import OrchestratorAgent


class TestDataEngineerAgent(unittest.TestCase):

    def setUp(self):
        # Create memory system and function registry
        self.memory = MemorySystem(persistence_path="./data/test_memory/")
        self.function_registry = FunctionRegistry()

        # Create agent
        self.data_engineer = DataEngineerAgent(self.memory, self.function_registry)

        # Create test dataframe
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, None, 40, 35],
            'value': [100, 200, 300, 400, 500]
        })

        self.data_engineer.current_df = self.test_df
        self.data_engineer.dataframes["main"] = self.test_df

    def test_handle_data_cleaning_missing_values(self):
        # Test checking for missing values
        result = self.data_engineer._handle_data_cleaning("Check for missing values")

        self.assertTrue(result["success"])
        self.assertIn("missing_values", result["result"])
        self.assertEqual(result["result"]["missing_values"]["age"]["count"], 1)

    def test_handle_data_cleaning_drop_na(self):
        # Test dropping rows with missing values
        original_shape = self.data_engineer.current_df.shape
        result = self.data_engineer._handle_data_cleaning("Drop rows with missing values")

        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["rows_removed"], 1)
        self.assertEqual(result["result"]["new_shape"][0], original_shape[0] - 1)

    def test_handle_data_cleaning_fill_na(self):
        # Test filling missing values
        result = self.data_engineer._handle_data_cleaning("Fill missing values with 0")

        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["values_filled"], 1)
        self.assertEqual(result["result"]["new_missing"], 0)

    def test_handle_data_transformation(self):
        # Test creating a new column
        result = self.data_engineer._handle_data_transformation("Create a new column 'double_value' = value * 2")

        self.assertTrue(result["success"])
        self.assertIn("double_value", self.data_engineer.current_df.columns)
        self.assertEqual(self.data_engineer.current_df["double_value"].iloc[0], 200)

    def test_handle_data_exploration(self):
        # Test data exploration
        result = self.data_engineer._handle_data_exploration("Show me a sample of 3 rows")

        self.assertTrue(result["success"])
        self.assertEqual(len(result["result"]["sample"]), 3)

        result = self.data_engineer._handle_data_exploration("Describe the data")

        self.assertTrue(result["success"])
        self.assertIn("description", result["result"])

    def tearDown(self):
        # Clean up any files created during tests
        import shutil
        if os.path.exists("./data/test_memory/"):
            shutil.rmtree("./data/test_memory/")


class TestVizSpecialistAgent(unittest.TestCase):

    def setUp(self):
        # Create memory system and function registry
        self.memory = MemorySystem(persistence_path="./data/test_memory/")
        self.function_registry = FunctionRegistry()

        # Create agent
        self.viz_specialist = VizSpecialistAgent(
            self.memory,
            self.function_registry,
            output_dir="./data/test_visualizations/"
        )

        # Create test dataframe
        self.test_df = pd.DataFrame({
            'id': range(100),
            'category': np.random.choice(['A', 'B', 'C'], size=100),
            'value1': np.random.normal(0, 1, size=100),
            'value2': np.random.normal(5, 2, size=100),
            'date': pd.date_range(start='2023-01-01', periods=100)
        })

    def test_extract_columns(self):
        # Test column extraction from query
        query = "Create a scatter plot with 'value1' and 'value2'"
        columns = self.viz_specialist._extract_columns(query, self.test_df)

        self.assertEqual(len(columns), 2)
        self.assertIn("value1", columns)
        self.assertIn("value2", columns)

    def test_create_histogram(self):
        # Test histogram creation
        result = self.viz_specialist._create_histogram("Create a histogram of value1", self.test_df)

        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["column"], "value1")
        self.assertTrue(os.path.exists(result["result"]["filepath"]))

    def test_create_scatter(self):
        # Test scatter plot creation
        result = self.viz_specialist._create_scatter(
            "Create a scatter plot of value1 vs value2",
            self.test_df
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["x_column"], "value1")
        self.assertEqual(result["result"]["y_column"], "value2")
        self.assertTrue(os.path.exists(result["result"]["filepath"]))

    def test_suggest_visualizations(self):
        # Test visualization suggestions
        result = self.viz_specialist._suggest_visualizations(
            "Suggest visualizations for value1, value2, and category",
            self.test_df
        )

        self.assertTrue(result["success"])
        self.assertGreater(len(result["result"]["suggestions"]), 0)

    def tearDown(self):
        # Clean up any files created during tests
        import shutil
        if os.path.exists("./data/test_memory/"):
            shutil.rmtree("./data/test_memory/")
        if os.path.exists("./data/test_visualizations/"):
            shutil.rmtree("./data/test_visualizations/")


class TestOrchestratorAgent(unittest.TestCase):

    def setUp(self):
        # Create memory system and function registry
        self.memory = MemorySystem(persistence_path="./data/test_memory/")
        self.function_registry = FunctionRegistry()
        self.conversation = ConversationManager(self.memory, persistence_path="./data/test_conversations/")

        # Create individual agents
        self.data_engineer = DataEngineerAgent(self.memory, self.function_registry)
        self.viz_specialist = VizSpecialistAgent(
            self.memory,
            self.function_registry,
            output_dir="./data/test_visualizations/"
        )
        self.memory_agent = MemoryAgent(self.memory, self.conversation)

        # Create orchestrator
        self.orchestrator = OrchestratorAgent(
            self.data_engineer,
            self.viz_specialist,
            self.memory_agent,
            self.conversation
        )

        # Create test dataframe
        self.test_df = pd.DataFrame({
            'id': range(100),
            'category': np.random.choice(['A', 'B', 'C'], size=100),
            'value1': np.random.normal(0, 1, size=100),
            'value2': np.random.normal(5, 2, size=100)
        })

        self.data_engineer.current_df = self.test_df
        self.data_engineer.dataframes["main"] = self.test_df

    def test_classify_query(self):
        # Test query classification
        data_query = "Clean the missing values in the dataset"
        self.assertEqual(self.orchestrator._classify_query(data_query), "data_processing")

        viz_query = "Create a histogram of value1"
        self.assertEqual(self.orchestrator._classify_query(viz_query), "visualization")

        memory_query = "Summarize what we've done so far"
        self.assertEqual(self.orchestrator._classify_query(memory_query), "memory")

    def test_process_data_query(self):
        # Test processing a data query
        result = self.orchestrator.process_query("Show me information about the columns")

        self.assertTrue(result["success"])
        self.assertIn("columns", result["result"])

    def test_process_viz_query(self):
        # Test processing a visualization query
        result = self.orchestrator.process_query("Create a histogram of value1")

        self.assertTrue(result["success"])

    def test_process_memory_query(self):
        # Test processing a memory query
        # First, add some operations to memory
        self.data_engineer._handle_data_exploration("Describe the data")
        self.data_engineer._handle_data_exploration("Show me a sample of 3 rows")

        result = self.orchestrator.process_query("Summarize what we've done so far")

        self.assertTrue(result["success"])

    def tearDown(self):
        # Clean up any files created during tests
        import shutil
        if os.path.exists("./data/test_memory/"):
            shutil.rmtree("./data/test_memory/")
        if os.path.exists("./data/test_visualizations/"):
            shutil.rmtree("./data/test_visualizations/")
        if os.path.exists("./data/test_conversations/"):
            shutil.rmtree("./data/test_conversations/")


if __name__ == '__main__':
    unittest.main()