# utils/autogen_utils.py

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_agent_configs() -> Dict[str, Any]:
    """Load agent configurations from yaml file"""
    config_path = Path("config/agent_config.yaml")
    try:
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        return configs
    except Exception as e:
        logger.error(f"Error loading agent configs: {e}")
        return {}


def validate_agent_response(response: Dict[str, Any]) -> bool:
    """Validate agent response format"""
    required_fields = ['status', 'message', 'data']
    return all(field in response for field in required_fields)


def format_agent_response(data: Any, message: str = "", status: str = "success") -> Dict[str, Any]:
    """Format agent response in standardized format"""
    return {
        "status": status,
        "message": message,
        "data": data,
        "timestamp": str(datetime.now())
    }


def create_agent_chain(agents: List[Dict[str, Any]]) -> List[Any]:
    """Create a chain of agents for sequential processing"""
    chain = []
    for agent_config in agents:
        agent_type = agent_config['type']
        agent_params = agent_config.get('params', {})
        agent = create_agent(agent_type, **agent_params)
        chain.append(agent)
    return chain


def create_agent(agent_type: str, **kwargs) -> Any:
    """Create an agent instance based on type"""
    agent_classes = {
        "data_engineer": DataEngineerAgentAuto,
        "viz_specialist": VizSpecialistAgentAuto,
        "memory_agent": MemoryAgentAuto,
        "orchestrator": OrchestratorAgentAuto
    }

    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent_classes[agent_type](**kwargs)