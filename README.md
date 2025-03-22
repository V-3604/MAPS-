# MAPS: Multi-Agent Processing System


## Overview
MAPS (Multi-Agent Processing System) is a conversational multi-agent framework for data cleaning, processing, and visualization. It maintains context across multiple interactions, providing an intuitive interface for performing data analysis tasks.

The system uses a specialized agent architecture where each agent handles specific aspects of the data analysis workflow, orchestrated by a central coordinator that routes user queries to the appropriate agent.


### Key Features
- Multi-Agent Architecture: Specialized agents for data engineering, visualization, and orchestration
- Context Preservation: Robust memory system with checkpoints, caching, and backup capabilities
- Data Transformation: Clean, process, and transform datasets using pandas
- Visualization: Create customized visualizations with support for multiple chart types
- Memory Management: Advanced memory system with compression, indexing, and query capabilities
- Workflow Engine: Support for sequential and conditional workflow execution
- Extensible Design: Register custom functions through the function registry


### System Architecture

#### Agent Roles

- Data Engineer Agent
  - Handles data loading, cleaning, transformation, and preprocessing
  - Executes pandas operations on datasets
  - Validates data against specified rules
  - Creates derived variables and features

- Visualization Specialist Agent
  - Creates visualizations using seaborn and matplotlib
  - Supports multiple visualization types (line, scatter, bar, histogram, heatmap, boxplot, pairplot, pie)
  - Customizes plots based on user requirements
  - Stores generated visualizations for future reference


- Orchestrator Agent
  - Routes requests to appropriate specialized agents
  - Coordinates the overall workflow
  - Maintains conversation flow and history
  - Presents results back to the user


#### Core Components


- Memory System
  - Sophisticated storage system with:
    - Data store with compression capabilities
    - Metadata tracking and indexing
    - Checkpoint and backup functionality
    - Cache management with TTL
    - Concurrent access with locking mechanisms
    - Query capability by metadata or content

- Function Registry
  - Centralized registry for:
    - Data processing functions (filter, aggregate, sort, etc.)
    - Visualization functions (line, scatter, bar, histogram, etc.)
    - Custom function registration

- Workflow Engine
    - Executes sequences of operations
    - Supports basic sequential workflows
    - Handles conditional execution paths
    - Passes data between workflow steps


### Current Capabilities

#### **The current implementation of MAPS can:**

- Load Data: Import data from pandas DataFrames and other sources
- Process Data: Filter, aggregate, sort, and transform data
- Validate Data: Check data against validation rules
- Visualize Data: Create multiple visualization types:
    - Line plots for time series or trends
    - Scatter plots for relationship analysis
    - Bar charts for categorical comparisons
    - Histograms for distribution analysis
    - Heatmaps for correlation visualization
    - Box plots for distribution comparison
    - Pair plots for multi-variable relationships
    - Pie charts for part-to-whole comparisons
- Execute Workflows: Run sequences of data operations and visualizations
- Manage Memory: Store, retrieve, and query data with advanced features
- Create Checkpoints: Save system state for later restoration
- Track Operations: Maintain history of all operations performed


### Limitations (Compared to Project Instructions)

- Natural Language Processing: The system doesn't yet parse natural language queries; it uses structured API requests
- Statistical Analysis: Limited built-in statistical analysis capabilities
- Interactive Visualizations: Visualizations are static; interactive plots aren't yet supported


### Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/MAPS.git
cd MAPS
```

2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Project Structure
```
MAPS/
├── agents/                 # Agent implementations
│   ├── data_engineer.py    # Data engineering agent
│   └── viz_specialist.py   # Visualization specialist agent
├── core/                   # Core system components
│   ├── memory_system.py    # Memory management system
│   ├── orchestrator.py     # Orchestrator agent
│   ├── workflow_engine.py  # Workflow execution engine
│   └── function_registry.py # Function registration system
├── data/                   # Data storage directories
│   ├── memory/             # Memory storage
│   ├── checkpoints/        # System checkpoints
│   ├── backups/            # System backups
│   ├── indexes/            # Search indexes
│   ├── cache/              # Cache storage
│   └── temp/               # Temporary files
├── output/                 # Output files
│   ├── logs/               # System logs
│   └── visualizations/     # Generated visualizations
├── utils/                  # Utility functions
│   └── data_validation.py  # Data validation utilities
└── test_run.py             # Example script to run the system
```


4. Example Usage

Here's how you might interact with the system in a Python script:

```
from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry
from core.orchestrator import Orchestrator
from agents.data_engineer import DataEngineer
from agents.viz_specialist import VizSpecialist

# Initialize components
memory_system = MemorySystem()
function_registry = FunctionRegistry()

# Register data processing functions
function_registry.register_function(data_filter, "data", "filter")
function_registry.register_function(data_aggregate, "data", "aggregate")
function_registry.register_function(data_sort, "data", "sort")

# Register visualization functions
function_registry.register_function(viz_line, "visualization", "viz_line")
function_registry.register_function(viz_scatter, "visualization", "viz_scatter")
function_registry.register_function(viz_bar, "visualization", "viz_bar")
function_registry.register_function(viz_histogram, "visualization", "viz_histogram")

# Initialize orchestrator with components
orchestrator = Orchestrator(
    memory_system=memory_system,
    function_registry=function_registry
)

# Process data
data_request = {
    "type": "data_processing",
    "operation": "load",
    "params": {
        "data": your_dataframe,
        "validation_rules": {
            "column_name": {
                "numeric": {},
                "range": {"min": 0, "max": 100}
            }
        }
    }
}
result = orchestrator.process_request(data_request)

# Create visualization
viz_request = {
    "type": "visualization",
    "viz_type": "scatter",
    "params": {
        "x": "column1",
        "y": "column2",
        "title": "Relationship between variables",
        "color": "category_column"
    }
}
result = orchestrator.process_request(viz_request)

# Execute workflow
workflow_request = {
    "type": "workflow",
    "workflow_type": "basic",
    "steps": [
        {
            "type": "data_operation",
            "operation": "filter",
            "params": {"column": "age", "condition": ">", "value": 30}
        },
        {
            "type": "visualization",
            "viz_type": "histogram",
            "params": {"x": "income", "bins": 20}
        }
    ]
}
result = orchestrator.process_request(workflow_request)
```


7. Running Tests
Run all tests:
```
python test_run.py
```


### Future Development
- Future enhancements planned for MAPS include:

  - Natural Language Processing: The system doesn't yet parse natural language queries; it uses structured API requests
  - Statistical Analysis: Limited built-in statistical analysis capabilities
  - Interactive Visualizations: Visualizations are static; interactive plots aren't yet supported

