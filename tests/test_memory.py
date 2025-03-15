import unittest
import sys
import os
import pandas as pd
import json
import pickle

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.memory_system import MemorySystem
from core.conversation_manager import ConversationManager


class TestMemorySystem(unittest.TestCase):

    def setUp(self):
        # Create memory system
        self.test_dir = "./data/test_memory/"

        # Create directory if it doesn't exist
        os.makedirs(self.test_dir, exist_ok=True)

        self.memory = MemorySystem(persistence_path=self.test_dir)

        # Create test dataframe
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'value': [100, 200, 300, 400, 500]
        })

    def test_update_data_state(self):
        # Test updating data state
        result = self.memory.update_data_state("test_file.csv", self.test_df)

        self.assertEqual(result["current_file"], "test_file.csv")
        self.assertEqual(result["shape"], (5, 4))
        self.assertEqual(len(result["dtypes"]), 4)

    def test_add_operation(self):
        # Test adding operations
        operation = self.memory.add_operation(
            "Test operation",
            "df.test_operation()",
            "Test result"
        )

        self.assertEqual(operation["step"], 1)
        self.assertEqual(operation["description"], "Test operation")
        self.assertEqual(operation["code"], "df.test_operation()")

        # Add another operation
        operation2 = self.memory.add_operation(
            "Second operation",
            "df.second_operation()",
            "Second result"
        )

        self.assertEqual(operation2["step"], 2)
        self.assertEqual(len(self.memory.memory["operation_history"]), 2)

    def test_add_visualization(self):
        # Test adding visualizations
        viz = self.memory.add_visualization(
            "histogram",
            "Test histogram",
            ["value"],
            "test_viz.png"
        )

        self.assertEqual(viz["type"], "histogram")
        self.assertEqual(viz["description"], "Test histogram")
        self.assertEqual(viz["columns"], ["value"])
        self.assertEqual(viz["file_path"], "test_viz.png")

    def test_summarize_current_context(self):
        # Add some operations and visualizations
        self.memory.add_operation("Op 1", "code1", "result1")
        self.memory.add_operation("Op 2", "code2", "result2")
        self.memory.add_visualization("type1", "desc1", ["col1"], "file1")

        # Test context summarization
        summary = self.memory.summarize_current_context()

        self.assertIn("recent_operations", summary)
        self.assertEqual(len(summary["recent_operations"]), 2)
        self.assertEqual(summary["visualization_count"], 1)

    def test_save_and_load_checkpoint(self):
        # Add some test data
        self.memory.update_data_state("test_file.csv", self.test_df)
        self.memory.add_operation("Test operation", "test_code", "test_result")
        self.memory.add_visualization("test_type", "test_desc", ["test_col"], "test_file.png")

        # Save checkpoint
        checkpoint_path = self.memory.save_checkpoint("test_checkpoint")

        # Create a new memory system
        new_memory = MemorySystem(persistence_path=self.test_dir)

        # Load checkpoint
        loaded_memory = new_memory.load_checkpoint(checkpoint_path)

        # Verify loaded data
        self.assertEqual(loaded_memory["data_state"]["current_file"], "test_file.csv")
        self.assertEqual(len(loaded_memory["operation_history"]), 1)
        self.assertEqual(len(loaded_memory["visualizations"]), 1)

    def test_save_and_load_dataframe(self):
        # Save dataframe
        file_path = self.memory.save_dataframe(self.test_df, "test_df.pkl")

        # Load dataframe
        loaded_df = self.memory.load_dataframe(file_path)

        # Verify loaded dataframe
        pd.testing.assert_frame_equal(loaded_df, self.test_df)

    def tearDown(self):
        # Clean up any files created during tests
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


class TestConversationManager(unittest.TestCase):

    def setUp(self):
        # Create memory system and conversation manager
        self.memory = MemorySystem(persistence_path="./data/test_memory/")
        self.test_dir = "./data/test_conversations/"

        # Create directory if it doesn't exist
        os.makedirs(self.test_dir, exist_ok=True)

        self.conversation = ConversationManager(self.memory, persistence_path=self.test_dir)

    def test_add_message(self):
        # Test adding a message
        message = self.conversation.add_message("user", "Test message")

        self.assertEqual(message["sender"], "user")
        self.assertEqual(message["content"], "Test message")
        self.assertEqual(message["session_id"], self.conversation.current_session_id)

    def test_get_session_history(self):
        # Add some messages
        self.conversation.add_message("user", "Message 1")
        self.conversation.add_message("system", "Response 1")
        self.conversation.add_message("user", "Message 2")

        # Get session history
        history = self.conversation.get_session_history()

        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["content"], "Message 1")
        self.assertEqual(history[1]["content"], "Response 1")
        self.assertEqual(history[2]["content"], "Message 2")

    def test_get_recent_history(self):
        # Add several messages
        for i in range(10):
            self.conversation.add_message("user", f"Message {i}")

        # Get recent history
        recent = self.conversation.get_recent_history(3)

        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0]["content"], "Message 7")
        self.assertEqual(recent[1]["content"], "Message 8")
        self.assertEqual(recent[2]["content"], "Message 9")

    def test_start_new_session(self):
        # Add a message to current session
        self.conversation.add_message("user", "Message in first session")
        first_session_id = self.conversation.current_session_id

        # Start a new session
        new_session_id = self.conversation.start_new_session()

        # Add a message to new session
        self.conversation.add_message("user", "Message in second session")

        # Check session histories
        first_session_history = self.conversation.get_session_history(first_session_id)
        second_session_history = self.conversation.get_session_history(new_session_id)

        self.assertEqual(len(first_session_history), 1)
        self.assertEqual(len(second_session_history), 1)
        self.assertEqual(first_session_history[0]["content"], "Message in first session")
        self.assertEqual(second_session_history[0]["content"], "Message in second session")

    def test_save_and_load_conversation(self):
        # Add some messages
        self.conversation.add_message("user", "Message 1")
        self.conversation.add_message("system", "Response 1")
        self.conversation.add_message("user", "Message 2")

        # Save conversation
        file_path = self.conversation.save_conversation("test_conversation.json")

        # Create new conversation manager
        new_conversation = ConversationManager(self.memory, persistence_path=self.test_dir)

        # Load conversation
        loaded_conversation = new_conversation.load_conversation(file_path)

        # Verify loaded data
        self.assertEqual(len(loaded_conversation), 3)
        self.assertEqual(loaded_conversation[0]["content"], "Message 1")
        self.assertEqual(loaded_conversation[1]["content"], "Response 1")
        self.assertEqual(loaded_conversation[2]["content"], "Message 2")

    def test_summarize_conversation(self):
        # Add some messages
        self.conversation.add_message("user", "What is the data about?")
        self.conversation.add_message("system", "The data contains sales information.")
        self.conversation.add_message("user", "Show me a summary of sales by region.")
        self.conversation.add_message("system", "Here's the summary of sales by region.")

        # Get conversation summary
        summary = self.conversation.summarize_conversation()

        self.assertIn("Conversation with", summary)
        self.assertIn("What is the data about?", summary)
        self.assertIn("Show me a summary", summary)

    def tearDown(self):
        # Clean up any files created during tests
        import shutil
        if os.path.exists("./data/test_memory/"):
            shutil.rmtree("./data/test_memory/")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()