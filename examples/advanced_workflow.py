import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry
from core.conversation_manager import ConversationManager

from agents.data_engineer import DataEngineerAgent
from agents.viz_specialist import VizSpecialistAgent
from agents.memory_agent import MemoryAgent
from agents.orchestrator import OrchestratorAgent

from config.system_config import DATA_DIRS, OUTPUT_DIRS
from utils.logging_utils import setup_logging, log_system_info
from utils.sample_data import create_customer_data


def run_advanced_workflow():
    """
    Run a more complex workflow to demonstrate the MAPS system capabilities.

    This function creates and connects all the agents,
    then processes a series of example queries demonstrating
    more advanced data processing and visualization.
    """
    print("Initializing MAPS system components...")

    # Set up logging
    logger = setup_logging({
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": os.path.join(OUTPUT_DIRS["logs"], "advanced_workflow.log"),
                "mode": "a"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": True
            }
        }
    })

    # Log system info
    log_system_info(logger)

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

    logger.info("System initialized. Processing example queries...")

    # Ensure we have the sample data file
    sample_file = os.path.join(DATA_DIRS["sample_datasets"], "customer_data.csv")
    if not os.path.exists(sample_file) or os.path.getsize(sample_file) == 0:
        sample_file = create_customer_data()

    # Example workflow with sequential queries
    example_queries = [
        # Data loading and exploration
        f"Load data from '{sample_file}'",
        "Show me information about the columns in this dataset",
        "What are the missing values in this dataset?",

        # Data cleaning and transformation
        "Drop rows where customer_id is missing",
        "Fill missing age values with the mean age",
        "Create a new column 'age_group' with categories 'Young' for < 30, 'Middle-aged' for 30-50, and 'Senior' for > 50",
        "Create a column 'total_spent' that adds the values from 'purchase_amount' and 'service_fee'",
        "Create a column 'is_loyal' that is True when membership_years > 2",

        # Basic visualizations
        "Show a histogram of age distribution",
        "Create a boxplot of total_spent by customer_segment",
        "Show a bar chart of average total_spent by age_group",

        # Advanced visualizations
        "Create a scatter plot of purchase_amount vs service_fee colored by customer_segment",
        "Show a heatmap of correlations between numeric columns",
        "Create a line plot of total_spent by registration_date",

        # Memory and context operations
        "Summarize what we've done so far",
        "Show all visualizations we've created",
        "Save our current state as a checkpoint",

        # Continued analysis
        "Group the data by age_group and customer_segment and show average total_spent",
        "Which customer segment has the highest average total_spent?",
        "Show me a visualization that compares total_spent across different customer segments and age groups"
    ]

    # Process each query
    results = []
    for i, query in enumerate(example_queries):
        print(f"\n[Query {i + 1}] {query}")
        logger.info(f"Processing query: {query}")

        response = orchestrator.process_query(query)

        print(f"[Response] {response['message']}")
        logger.info(f"Response: {response['message']}")

        results.append(response)

    print("\nAdvanced workflow completed successfully.")
    logger.info("Advanced workflow completed successfully.")

    return results


if __name__ == "__main__":
    # Run the example workflow
    results = run_advanced_workflow()