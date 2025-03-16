# MAPS: Multi-Agent Processing System


## Overview
MAPS (Multi-Agent Processing System) is a conversational multi-agent framework for data cleaning, processing, and visualization. It maintains context across multiple interactions, providing an intuitive natural language interface for performing data analysis tasks.

The system uses a specialized agent architecture where each agent handles specific aspects of the data analysis workflow, orchestrated by a central coordinator that routes user queries to the appropriate agent.


### Key Features
- Natural Language Interface: Process data operations through conversational queries
- Multi-Agent Architecture: Specialized agents for data engineering, visualization, and memory management
- Context Preservation: Maintains memory of operations and data state across conversations
- Data Transformation: Clean, process, and transform datasets using pandas
- Visualization: Create customized visualizations based on data characteristics
- Memory Management: Save and retrieve context and operation history
- Extensible Design: Easy to add new capabilities through the agent framework


### System Architecture

#### Agent Roles

- Data Engineer Agent
  - Handles data loading, cleaning, transformation, and preprocessing
  - Executes pandas operations on datasets
  - Maintains data state and tracks transformations
  - Creates derived variables and features


- Visualization Specialist Agent
  - Creates visualizations using seaborn and matplotlib
  = Recommends appropriate visualization types based on data
  - Stores generated visualizations for future reference
  - Customizes plots based on user requirements
  - Memory & Context Agent
  - Maintains summary of operations performed
  - Tracks conversation history and session information
  - Creates checkpoints of data state
  - Retrieves historical context when needed


- Orchestrator Agent
  - Routes queries to appropriate specialized agents
  - Coordinates the overall workflow
  - Maintains conversation flow
  - Presents results back to the user


#### Core Components


- Memory System
  - The system maintains a structured memory that includes:
    - Current data state (path, shape, columns, data types)
    - History of operations performed
    - Key variables and their purposes
    - Generated visualizations

- Function Registry
  - Maintains collections of:
    - Data processing functions
    - Visualization functions
    - Utility functions

- Conversation Manager
  - Tracks conversation across multiple sessions
  - Maintains message history
  - Provides context for operations


### Current Capabilities

#### **The current implementation of MAPS can:**

- Load Data: Import data from CSV files
- Explore Data: Show data samples, column information, missing values, and descriptive statistics
- Clean Data: Drop or fill missing values, filter rows, handle duplicates
- Transform Data: Create new columns with computed values, modify existing columns
- Visualize Data: Create common plot types including:
  - Histograms for distribution analysis
  - Bar charts for categorical comparisons
  - Scatter plots for relationship exploration
  - Boxplots for distribution comparison
  - Heatmaps for correlation analysis
- Track Context: Maintain memory of operations across conversations
- Save Progress: Create checkpoints of the current state
- Summarize Work: Provide summaries of operations performed


### Limitations (Compared to Project Instructions)

#### **While the system meets most requirements from the project specifications, there are some limitations:**

- Integration with AutoGen/CrewAI: The current implementation uses a custom agent framework rather than AutoGen or CrewAI.
- Function Dictionary Implementation: Instead of explicit dictionaries as shown in the project plan, functions are registered through a more flexible registry system.
- Advanced NLP Understanding: Query parsing is currently handled through basic pattern matching rather than sophisticated NLP methods.
- Visualization Recommendations: While the system can create appropriate visualizations, the capability to proactively recommend optimal visualization types is limited.
- Error Recovery: Error handling exists but could be more robust for complex scenarios.


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
├── agents/               # Agent implementations
├── config/               # Configuration files
├── core/                 # Core system components
├── data/                 # Data storage
│   ├── memory/           # Memory checkpoints
│   ├── conversations/    # Conversation history
│   ├── visualizations/   # Generated visualizations
│   └── sample_datasets/  # Sample data for examples
├── examples/             # Example scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
└── utils/                # Utility functions
```

4. Getting Started:
- Run the basic workflow example:
```
python examples/basic_workflow.py
```
- Or explore the advanced workflow:
```
python examples/advanced_workflow.py
```

5. Using in a Notebook
- Check out the example notebooks in the notebooks/ directory:
  - getting_started.ipynb: Basic introduction to using MAPS
  - example_workflows.ipynb: More complex analysis workflows

6. Example Usage

Here's how you might interact with the system in a Python script:

```
from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry
from core.conversation_manager import ConversationManager
from agents.data_engineer import DataEngineerAgent
from agents.viz_specialist import VizSpecialistAgent
from agents.memory_agent import MemoryAgent
from agents.orchestrator import OrchestratorAgent
from config.system_config import DATA_DIRS

# Initialize components
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

# Process queries
orchestrator.process_query("Load data from 'data/sample_datasets/sales_data.csv'")
orchestrator.process_query("Show me the first 5 rows of data")
orchestrator.process_query("Create a new column 'total_revenue' as price * quantity")
orchestrator.process_query("Show a histogram of total_revenue")
orchestrator.process_query("Summarize what we've done so far")
```


7. Running Tests
Run all tests:
```
python -m unittest discover
```
Or run specific test files:
```
python -m unittest tests.test_agents
python -m unittest tests.test_memory
python -m unittest tests.test_workflows
```
### Future Development
- Future enhancements planned for MAPS include:
  - Enhanced Natural Language Understanding: Improve query parsing with more sophisticated NLP methods
  - Additional Agent Types: Add specialized agents for statistical analysis, time series analysis, etc.
  - Database Integration: Add support for SQL databases and other data sources
  - Web Interface: Create a user-friendly web interface for interactive sessions
  - Report Generation: Automatically generate reports summarizing analyses
  - Advanced Visualization: Support more complex and interactive visualizations
  - Model Integration: Add machine learning model building and evaluation capabilities

