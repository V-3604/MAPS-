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


class TestBasicWorkflow(unittest.TestCase):

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
            'price': np.random.uniform(10, 1000, size=100),
            'quantity': np.random.randint(1, 50, size=100),
            'date': pd.date_range(start='2023-01-01', periods=100)
        })

        # Save test dataframe for loading
        os.makedirs("./data/test_datasets/", exist_ok=True)
        self.test_file = "./data/test_datasets/test_data.csv"
        self.test_df.to_csv(self.test_file, index=False)

    def test_full_workflow(self):
        """Test a complete workflow with multiple sequential operations"""
        # Define a sequence of queries to test
        queries = [
            f"Load data from '{self.test_file}'",
            "Show me the first 5 rows of data",
            "Create a new column 'total' as price times quantity",
            "Show a histogram of the total column",
            "Create a scatter plot of price vs quantity",
            "Summarize what we've done so far",
            "Save a checkpoint of our current state"
        ]

        # Process each query
        results = []
        for query in queries:
            result = self.orchestrator.process_query(query)
            results.append(result)

            # Assert each operation was successful
            self.assertTrue(result["success"], f"Query failed: {query}")

        # Verify the data has been transformed as expected
        self.assertIn("total", self.data_engineer.current_df.columns)

        # Verify visualizations were created
        self.assertGreaterEqual(len(self.memory.memory["visualizations"]), 2)

        # Verify operations history contains all operations
        self.assertGreaterEqual(len(self.memory.memory["operation_history"]), len(queries))

        # Verify conversation history contains all interactions
        session_history = self.conversation.get_session_history()
        self.assertEqual(len(session_history), len(queries) * 2)  # Query + response for each

    def test_error_recovery(self):
        """Test the system's ability to recover from errors"""
        # First load data
        self.orchestrator.process_query(f"Load data from '{self.test_file}'")

        # Then try an operation that should cause an error
        error_query = "Create a new column 'invalid' as nonexistent_column * 2"
        error_result = self.orchestrator.process_query(error_query)

        # Verify error was handled
        self.assertFalse(error_result["success"])

        # Try a valid operation after the error
        recovery_query = "Create a new column 'valid_column' as price * 2"
        recovery_result = self.orchestrator.process_query(recovery_query)

        # Verify system recovered
        self.assertTrue(recovery_result["success"])
        self.assertIn("valid_column", self.data_engineer.current_df.columns)

    def test_context_retention(self):
        """Test the system's ability to retain context across operations"""
        # Load data and create a new column
        self.orchestrator.process_query(f"Load data from '{self.test_file}'")
        self.orchestrator.process_query("Create a new column 'total' as price times quantity")

        # Request a visualization using the new column
        viz_result = self.orchestrator.process_query("Show a histogram of total")

        # Verify visualization succeeded
        self.assertTrue(viz_result["success"])

        # Ask about operations history
        memory_result = self.orchestrator.process_query("What have we done so far?")

        # Verify memory response includes information about all operations
        self.assertTrue(memory_result["success"])

    def tearDown(self):
        # Clean up any files created during tests
        import shutil
        if os.path.exists("./data/test_memory/"):
            shutil.rmtree("./data/test_memory/")
        if os.path.exists("./data/test_visualizations/"):
            shutil.rmtree("./data/test_visualizations/")
        if os.path.exists("./data/test_conversations/"):
            shutil.rmtree("./data/test_conversations/")
        if os.path.exists("./data/test_datasets/"):
            shutil.rmtree("./data/test_datasets/")


if __name__ == '__main__':
    unittest.main()