# basic_workflow.py

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path
from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry
import json
import uuid


class BasicWorkflow:
    def __init__(self, memory_system: MemorySystem, function_registry: FunctionRegistry):
        self.memory_system = memory_system
        self.function_registry = function_registry
        self.workflow_history = []
        self.current_workflow = None
        self.workflow_states = {}
        self.error_handlers = {}
        self.logger = self._setup_logger()
        self.setup_directories()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('output/logs/basic_workflow.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "data/workflows",
            "data/workflows/checkpoints",
            "data/workflows/temp",
            "output/workflows"
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def create_workflow(self, steps: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new workflow with steps and metadata"""
        try:
            workflow_id = str(uuid.uuid4())
            workflow = {
                "id": workflow_id,
                "steps": steps,
                "metadata": metadata or {},
                "status": "created",
                "creation_time": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "executed_steps": [],
                "current_step": None
            }

            # Validate workflow before storing
            validation = self.validate_workflow(steps)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": "Invalid workflow",
                    "validation_issues": validation["issues"]
                }

            self.workflow_states[workflow_id] = workflow
            self.logger.info(f"Created workflow: {workflow_id}")

            return {
                "success": True,
                "workflow_id": workflow_id,
                "message": "Workflow created successfully"
            }

        except Exception as e:
            self.logger.error(f"Error creating workflow: {str(e)}")
            return {"success": False, "error": str(e)}

    def execute_workflow(self, workflow_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow by ID"""
        try:
            workflow = self.workflow_states.get(workflow_id)
            if not workflow:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}

            workflow["status"] = "running"
            workflow["start_time"] = datetime.now().isoformat()
            self.current_workflow = workflow

            results = []
            for i, step in enumerate(workflow["steps"]):
                workflow["current_step"] = i

                # Update step with global params if provided
                if params:
                    step_params = step.get("params", {})
                    step_params.update(params)
                    step["params"] = step_params

                step_result = self._execute_step(step)
                results.append(step_result)
                workflow["executed_steps"].append({
                    "step_index": i,
                    "step_data": step,
                    "result": step_result,
                    "execution_time": datetime.now().isoformat()
                })

                if not step_result["success"] and step.get("critical", False):
                    workflow["status"] = "failed"
                    self._handle_workflow_failure(workflow, step, step_result)
                    break

            workflow["status"] = "completed" if all(r["success"] for r in results) else "failed"
            workflow["end_time"] = datetime.now().isoformat()
            workflow["results"] = results

            self.workflow_history.append(workflow)
            self._save_workflow_results(workflow)

            return {
                "success": workflow["status"] == "completed",
                "workflow_id": workflow_id,
                "results": results,
                "status": workflow["status"]
            }

        except Exception as e:
            self.logger.error(f"Error executing workflow: {str(e)}")
            if workflow:
                workflow["status"] = "failed"
                workflow["error"] = str(e)
            return {"success": False, "error": str(e)}

    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            step_type = step["type"]
            params = step.get("params", {})

            # Get the appropriate function based on step type
            function_category = {
                "data_operation": "data",
                "visualization": "visualization",
                "utility": "utility"
            }.get(step_type)

            if not function_category:
                raise ValueError(f"Unknown step type: {step_type}")

            func = self.function_registry.get_function(step["function"], function_category)
            if not func:
                raise ValueError(f"Function not found: {step['function']}")

            # Execute function and capture result
            result = func(**params)

            # Log execution in memory system
            self.memory_system.add_operation(
                operation_type=f"workflow_step_{step_type}",
                details=step,
                result=result
            )

            return {"success": True, "result": result}

        except Exception as e:
            self.logger.error(f"Step execution error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_workflow_failure(self, workflow: Dict[str, Any], failed_step: Dict[str, Any],
                                 failure_result: Dict[str, Any]):
        """Handle workflow failure"""
        try:
            error_handler = self.error_handlers.get(workflow["id"])
            if error_handler:
                error_handler(workflow, failed_step, failure_result)

            # Create failure checkpoint
            checkpoint_path = f"data/workflows/checkpoints/failure_{workflow['id']}_{datetime.now().isoformat()}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    "workflow": workflow,
                    "failed_step": failed_step,
                    "failure_result": failure_result,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

            self.logger.error(f"Workflow {workflow['id']} failed at step {failed_step.get('function')}")

        except Exception as e:
            self.logger.error(f"Error handling workflow failure: {str(e)}")

    def _save_workflow_results(self, workflow: Dict[str, Any]):
        """Save workflow results to file"""
        try:
            output_path = f"output/workflows/{workflow['id']}_{datetime.now().isoformat()}.json"
            with open(output_path, 'w') as f:
                json.dump(workflow, f, indent=2)

            self.logger.info(f"Saved workflow results to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving workflow results: {str(e)}")

    def validate_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate workflow steps before execution"""
        try:
            issues = []
            for i, step in enumerate(workflow_steps):
                if "type" not in step:
                    issues.append(f"Step {i}: Missing 'type' field")
                    continue

                if "function" not in step:
                    issues.append(f"Step {i}: Missing 'function' field")
                    continue

                # Validate function existence and parameters
                function_category = {
                    "data_operation": "data",
                    "visualization": "visualization",
                    "utility": "utility"
                }.get(step["type"])

                if not function_category:
                    issues.append(f"Step {i}: Invalid step type '{step['type']}'")
                    continue

                func = self.function_registry.get_function(step["function"], function_category)
                if not func:
                    issues.append(f"Step {i}: Function '{step['function']}' not found in {function_category} category")
                    continue

                # Validate parameters if provided
                if "params" in step:
                    param_validation = self.function_registry.validate_function(
                        func,
                        expected_params=list(step["params"].keys())
                    )
                    if not param_validation["valid"]:
                        issues.extend([f"Step {i}: {issue}" for issue in param_validation["issues"]])

            return {
                "valid": len(issues) == 0,
                "issues": issues
            }

        except Exception as e:
            self.logger.error(f"Error validating workflow: {str(e)}")
            return {"valid": False, "issues": [str(e)]}

    def register_error_handler(self, workflow_id: str, handler: callable) -> Dict[str, Any]:
        """Register an error handler for a workflow"""
        try:
            self.error_handlers[workflow_id] = handler
            return {"success": True, "message": f"Error handler registered for workflow {workflow_id}"}
        except Exception as e:
            self.logger.error(f"Error registering error handler: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        try:
            workflow = self.workflow_states.get(workflow_id)
            if not workflow:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}

            return {
                "success": True,
                "status": workflow["status"],
                "current_step": workflow["current_step"],
                "executed_steps": len(workflow["executed_steps"]),
                "total_steps": len(workflow["steps"]),
                "start_time": workflow.get("start_time"),
                "end_time": workflow.get("end_time")
            }

        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            return {"success": False, "error": str(e)}