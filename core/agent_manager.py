# core/agent_manager.py

from typing import Dict, Any
from agents.autogen_agents import (
    DataEngineerAgentAuto,
    VizSpecialistAgentAuto,
    MemoryAgentAuto,
    OrchestratorAgentAuto
)

class AgentManager:
    def __init__(self, memory_system, function_registry):
        self.memory_system = memory_system
        self.function_registry = function_registry
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agent types with necessary components"""
        self.agents = {
            "orchestrator": OrchestratorAgentAuto(
                name="Orchestrator",
                memory_system=self.memory_system,
                function_registry=self.function_registry
            ),
            "data_engineer": DataEngineerAgentAuto(
                name="DataEngineer",
                memory_system=self.memory_system,
                function_registry=self.function_registry
            ),
            "viz_specialist": VizSpecialistAgentAuto(
                name="VizSpecialist",
                memory_system=self.memory_system,
                function_registry=self.function_registry
            ),
            "memory_agent": MemoryAgentAuto(
                name="MemoryAgent",
                memory_system=self.memory_system,
                function_registry=self.function_registry
            )
        }

    def get_agent(self, agent_type: str):
        """Get a specific agent by type"""
        return self.agents.get(agent_type)

    def process_query(self, query: str):
        """Process a query through the orchestrator"""
        orchestrator = self.agents["orchestrator"]
        return orchestrator.process_message(query, sender="user")

    def reset_agents(self):
        """Reset all agents to initial state"""
        self._initialize_agents()