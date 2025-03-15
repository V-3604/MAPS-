import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry
from core.conversation_manager import ConversationManager

from agents.data_engineer import DataEngineerAgent
from agents.viz_specialist import VizSpecialistAgent
from agents.memory_agent import MemoryAgent
from agents.orchestrator import OrchestratorAgent

from config.system_config import DATA_DIRS
from utils.sample_data import create_sales_data


def run_basic_workflow():
    """
    Run a basic workflow to demonstrate the MAPS system.

    This function creates and connects all the agents,
    then processes a series of example queries.
    """
    print("Initializing MAPS system components...")

    # Initialize core components
    memory_system = MemorySystem(persistence_path=DATA_DIRS["memory"])
    function_registry = FunctionRegistry()
    conversation_manager = ConversationManager(memory_system, persistence_path=DATA_DIRS["conversations"])

    # Initialize agents
    data_engineer = DataEngineerAgent(memory_system, function_registry)
    viz_specialist = VizSpecialistAgent(
        memory_system,
        function_registry,
        output_dir=DATA_DIRS["visualizations"]
    )
    memory_agent = MemoryAgent(memory_system, conversation_manager)

    # Initialize orchestrator
    orchestrator = OrchestratorAgent(
        data_engineer,
        viz_specialist,
        memory_agent,
        conversation_manager
    )

    print("System initialized. Processing example queries...")

    # Ensure we have the sample data file
    sample_file = os.path.join(DATA_DIRS["sample_datasets"], "sales_data.csv")
    if not os.path.exists(sample_file) or os.path.getsize(sample_file) == 0:
        sample_file = create_sales_data()

    # Example workflow with sequential queries
    example_queries = [
        f"Load data from '{sample_file}'",
        "Show me the first 5 rows of data",
        "Check for missing values in the data",
        "Drop rows with missing values",
        "Create a new column 'total_revenue' as price * quantity",
        "Show the distribution of total_revenue",
        "Create a bar chart of average total_revenue by region",
        "Show a scatter plot of price vs quantity",
        "Summarize what we've done so far",
        "Save a checkpoint of our current state"
    ]

    # Process each query
    results = []
    for i, query in enumerate(example_queries):
        print(f"\n[Query {i + 1}] {query}")

        response = orchestrator.process_query(query)

        print(f"[Response] {response['message']}")
        results.append(response)

    print("\nWorkflow completed successfully.")
    return results


if __name__ == "__main__":
    # Run the example workflow
    results = run_basic_workflow()