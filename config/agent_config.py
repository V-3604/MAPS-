# Agent configuration settings
agent_configs = {
    "data_engineer": {
        "name": "Data Engineer",
        "description": "Handles data cleaning, transformation, and preprocessing",
        "capabilities": [
            "Loading data from various sources",
            "Cleaning data (handling missing values, duplicates)",
            "Transforming data (creating new columns, aggregations)",
            "Data type conversions and standardization",
            "Basic data exploration and statistics"
        ]
    },
    "viz_specialist": {
        "name": "Visualization Specialist",
        "description": "Creates visualizations using seaborn, matplotlib, or other libraries",
        "capabilities": [
            "Creating various types of plots (histogram, scatter, bar, box, etc.)",
            "Suggesting appropriate visualizations for given data",
            "Customizing plot appearance for clarity",
            "Saving visualizations for future reference",
            "Explaining visualization insights"
        ]
    },
    "memory_agent": {
        "name": "Memory & Context Agent",
        "description": "Maintains context across conversations and tracks system state",
        "capabilities": [
            "Tracking operation history",
            "Maintaining data state information",
            "Summarizing context for users and other agents",
            "Creating checkpoints of system state",
            "Retrieving past operations and visualizations"
        ]
    },
    "orchestrator": {
        "name": "Orchestrator Agent",
        "description": "Coordinates between agents based on user requests",
        "capabilities": [
            "Classifying user queries by intent",
            "Routing tasks to appropriate specialized agents",
            "Maintaining conversation flow",
            "Ensuring context is preserved across interactions",
            "Providing help and guidance to users"
        ]
    }
}

# Settings for conversation management
conversation_settings = {
    "max_history_length": 100,  # Maximum number of messages to keep in history
    "summary_interval": 10,     # Create summaries every N messages
    "checkpoint_interval": 10   # Auto-save checkpoints every N operations
}

# Settings for memory system
memory_settings = {
    "max_operations_history": 100,  # Maximum number of operations to keep in history
    "auto_checkpoint": True,        # Whether to automatically create checkpoints
    "checkpoint_interval": 10       # Create checkpoints every N operations
}

# Settings for visualization
visualization_settings = {
    "default_figsize": (10, 6),
    "default_dpi": 100,
    "save_format": "png",
    "style": "whitegrid",
    "context": "notebook",
    "color_palette": "deep"
}