# core/orchestrator.py

from typing import Dict, Any, List, Optional
import logging
import uuid
import json
from datetime import datetime
from pathlib import Path
from core.memory_system import MemorySystem
from core.workflow_engine import WorkflowEngine
from core.function_registry import FunctionRegistry
from agents.data_engineer import DataEngineer
from agents.viz_specialist import VizSpecialist
from utils.data_validation import DataValidator


class Orchestrator:
    """
    Orchestrates interactions between components and agents
    """

    def __init__(self,
                 memory_system: Optional[MemorySystem] = None,
                 function_registry: Optional[FunctionRegistry] = None):
        """
        Initialize the orchestrator

        Parameters:
        -----------
        memory_system : MemorySystem, optional
            Memory system instance
        function_registry : FunctionRegistry, optional
            Function registry instance
        """
        # Initialize components
        self.memory_system = memory_system or MemorySystem()
        self.function_registry = function_registry or FunctionRegistry()
        self.workflow_engine = WorkflowEngine(self.memory_system, self.function_registry)

        # Initialize specialized agents
        self.data_validator = DataValidator()

        # Set up agent management
        self.active_agents = {
            "data_engineer": DataEngineer(self.memory_system),
            "viz_specialist": VizSpecialist(self.memory_system, self.function_registry)
        }

        # Set up request processing
        self.request_handlers = {
            "data_processing": self._handle_data_processing,
            "visualization": self._handle_visualization,
            "workflow": self._handle_workflow,
            "memory": self._handle_memory,
            "agent_management": self._handle_agent_management,
            "system_config": self._handle_system_config
        }

        # Set up logging
        self.logger = self._setup_logger()
        self.conversation_history = []

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        Path('output/logs').mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler('output/logs/orchestrator.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request and route it to the appropriate handler

        Parameters:
        -----------
        request : Dict[str, Any]
            Request parameters

        Returns:
        --------
        Dict[str, Any]
            Operation result
        """
        try:
            # Add request ID if not present
            if "request_id" not in request:
                request["request_id"] = str(uuid.uuid4())

            # Extract request type
            request_type = request.get("type")

            # Log the request
            self.logger.info(f"Processing request: {request_type} (ID: {request['request_id']})")

            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "request": request,
                "type": request_type,
                "request_id": request["request_id"]
            })

            # Route to appropriate handler
            if request_type in self.request_handlers:
                result = self.request_handlers[request_type](request)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown request type: {request_type}"
                }

            # Record the result
            self.conversation_history[-1]["result"] = result

            return result
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_data_processing(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data processing requests"""
        try:
            data_engineer = self.active_agents.get("data_engineer")
            if not data_engineer:
                return {"success": False, "error": "Data Engineer agent not initialized"}

            operation = request.get("operation")
            params = request.get("params", {})

            if operation == "load":
                result = data_engineer.load_data(params.get("data"))

                # Validate data if rules provided
                if "validation_rules" in params:
                    validation_result = self.data_validator.validate_dataset(
                        data_engineer.current_dataset,
                        params["validation_rules"]
                    )

                    result["validation_summary"] = validation_result["summary"]
                    result["message"] = "Data loaded and validated successfully"

                    # Store validation result in memory
                    self.memory_system.add_operation(
                        operation_type="data_validation",
                        details={"rules": params["validation_rules"]},
                        result=validation_result
                    )

                return result

            elif operation == "transform":
                result = data_engineer.transform_data(params.get("operations", []))

                # Validate data if rules provided
                if "validation_rules" in params:
                    validation_result = self.data_validator.validate_dataset(
                        data_engineer.current_dataset,
                        params["validation_rules"]
                    )

                    result["validation_result"] = validation_result

                return result

            elif operation == "save":
                return data_engineer.save_data(
                    params.get("file_path", "output/data/processed_data.csv"),
                    params.get("format")
                )

            else:
                return {"success": False, "error": f"Unknown data operation: {operation}"}

        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_visualization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle visualization requests"""
        try:
            viz_specialist = self.active_agents.get("viz_specialist")
            if not viz_specialist:
                return {"success": False, "error": "Visualization Specialist agent not initialized"}

            viz_type = request.get("viz_type")
            params = request.get("params", {})

            # Get the current data from memory if not provided in params
            if "data" not in params:
                # Try to get the dataset from memory
                data_result = self.memory_system.handle_memory_request({
                    "operation": "retrieve",
                    "params": {"key": "test_data"}
                })

                if data_result.get("success", False):
                    params["data"] = data_result.get("data")

            result = viz_specialist.create_visualization(viz_type, params)

            # Store visualization result in memory system
            self.memory_system.add_operation(
                operation_type="visualization_create",
                details={"type": viz_type, "params": params},
                result=result
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in visualization: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_workflow(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow requests"""
        try:
            workflow_type = request.get("workflow_type", "basic")
            steps = request.get("steps", [])
            params = request.get("params", {})

            if workflow_type == "basic":
                return self.workflow_engine.execute_basic_workflow(steps, params)
            elif workflow_type == "advanced":
                return self.workflow_engine.execute_advanced_workflow(steps, params)
            else:
                return {"success": False, "error": f"Unknown workflow type: {workflow_type}"}

        except Exception as e:
            self.logger.error(f"Error in workflow: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory operation requests"""
        try:
            return self.memory_system.handle_memory_request(request)
        except Exception as e:
            self.logger.error(f"Error in memory operation: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_agent_management(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent management requests"""
        try:
            operation = request.get("operation")
            agent_id = request.get("agent_id")

            if operation == "status":
                # Get status of all agents or a specific agent
                if agent_id:
                    if agent_id in self.active_agents:
                        return {
                            "success": True,
                            "status": {
                                agent_id: self._get_agent_status(agent_id)
                            }
                        }
                    else:
                        return {"success": False, "error": f"Unknown agent: {agent_id}"}
                else:
                    # Get status of all agents
                    return {
                        "success": True,
                        "status": {
                            agent_id: self._get_agent_status(agent_id)
                            for agent_id in self.active_agents
                        }
                    }
            elif operation == "configure":
                # Configure an agent
                if not agent_id:
                    return {"success": False, "error": "No agent specified for configuration"}

                if agent_id not in self.active_agents:
                    return {"success": False, "error": f"Unknown agent: {agent_id}"}

                config = request.get("config", {})
                agent = self.active_agents[agent_id]

                if hasattr(agent, "configure") and callable(agent.configure):
                    return agent.configure(config)
                else:
                    return {"success": False, "error": f"Agent {agent_id} does not support configuration"}
            else:
                return {"success": False, "error": f"Unknown agent operation: {operation}"}

        except Exception as e:
            self.logger.error(f"Error in agent management: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_system_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system configuration requests"""
        try:
            config_type = request.get("config_type")
            updates = request.get("updates", {})

            if config_type == "memory":
                return self.memory_system.update_config(updates.get("memory", {}))
            elif config_type == "workflow":
                return self.workflow_engine.update_config(updates.get("workflow", {}))
            elif config_type == "function_registry":
                return self.function_registry.update_config(updates.get("function_registry", {}))
            elif config_type == "system":
                # Handle global system configuration
                results = []

                # Apply configurations to each component
                if "memory" in updates:
                    results.append(self.memory_system.update_config(updates["memory"]))

                if "workflow" in updates:
                    results.append(self.workflow_engine.update_config(updates["workflow"]))

                if "function_registry" in updates:
                    results.append(self.function_registry.update_config(updates["function_registry"]))

                return {"success": True, "message": "System configuration updated successfully"}
            else:
                return {"success": False, "error": f"Unknown configuration type: {config_type}"}

        except Exception as e:
            self.logger.error(f"Error in system configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status information for an agent"""
        agent = self.active_agents.get(agent_id)

        if not agent:
            return {"status": "inactive"}

        # Get status if the agent has a get_status method
        if hasattr(agent, "get_status") and callable(agent.get_status):
            return agent.get_status()
        else:
            # Basic status
            return {
                "status": "active",
                "queue_size": 0  # Placeholder
            }

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation history"""
        try:
            if not self.conversation_history:
                return {
                    "success": True,
                    "summary": {
                        "total_interactions": 0,
                        "last_interaction": None,
                        "active_agents": list(self.active_agents.keys()),
                        "current_context": None
                    }
                }

            return {
                "success": True,
                "summary": {
                    "total_interactions": len(self.conversation_history),
                    "last_interaction": self.conversation_history[-1],
                    "active_agents": list(self.active_agents.keys()),
                    "current_context": {
                        "last_request": self.conversation_history[-1]["request"],
                        "timestamp": datetime.now().isoformat(),
                        "request_id": self.conversation_history[-1]["request_id"]
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating conversation summary: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_state(self, file_path: str) -> Dict[str, Any]:
        """Save the current state to a file"""
        try:
            # Create state object
            state = {
                "timestamp": datetime.now().isoformat(),
                "conversation_history_length": len(self.conversation_history),
                "last_request_id": self.conversation_history[-1]["request_id"] if self.conversation_history else None,
                "active_agents": list(self.active_agents.keys())
            }

            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)

            return {
                "success": True,
                "message": f"State saved to {file_path}"
            }
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            return {"success": False, "error": str(e)}