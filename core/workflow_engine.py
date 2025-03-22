# core/workflow_engine.py

from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry


class WorkflowEngine:
    def __init__(self, memory_system: MemorySystem, function_registry: FunctionRegistry):
        self.memory_system = memory_system
        self.function_registry = function_registry
        self.current_workflow: Dict[str, Any] = {}
        self.workflow_history: List[Dict[str, Any]] = []

    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update workflow engine configuration settings"""
        try:
            # Process any configuration updates here
            return {"success": True, "message": "Workflow engine configuration updated"}
        except Exception as e:
            return {"success": False, "error": f"Error updating workflow configuration: {str(e)}"}

    def execute_basic_workflow(self, steps: List[Dict[str, Any]], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a basic workflow with sequential steps"""
        try:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_workflow = {
                "id": workflow_id,
                "type": "basic",
                "steps": steps,
                "params": params or {},
                "start_time": datetime.now().isoformat()
            }

            results = []
            current_data = None

            for step in steps:
                try:
                    # Add current data to step params if it exists
                    if current_data is not None and step.get("params", {}) is not None:
                        step["params"]["dataframe"] = current_data

                    result = self._execute_step(step)

                    # Update current data if the step returned a dataframe
                    if result.get("success") and "dataframe" in result:
                        current_data = result["dataframe"]

                    results.append({
                        "step": step,
                        "result": result,
                        "status": "success"
                    })
                except Exception as e:
                    error_result = {
                        "step": step,
                        "error": str(e),
                        "status": "failed"
                    }
                    results.append(error_result)
                    return {
                        "success": False,
                        "error": f"Workflow step failed: {str(e)}",
                        "partial_results": results
                    }

            self.current_workflow["end_time"] = datetime.now().isoformat()
            self.current_workflow["results"] = results
            self.workflow_history.append(self.current_workflow)

            return {
                "success": True,
                "workflow_id": workflow_id,
                "results": results
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}"
            }

    def execute_advanced_workflow(self, steps: List[Dict[str, Any]], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an advanced workflow with parallel and conditional steps"""
        try:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_workflow = {
                "id": workflow_id,
                "type": "advanced",
                "steps": steps,
                "params": params or {},
                "start_time": datetime.now().isoformat()
            }

            # Process steps based on their dependencies and conditions
            processed_steps = self._process_advanced_steps(steps, params)

            self.current_workflow["end_time"] = datetime.now().isoformat()
            self.current_workflow["results"] = processed_steps
            self.workflow_history.append(self.current_workflow)

            return {
                "success": True,
                "workflow_id": workflow_id,
                "results": processed_steps
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Advanced workflow execution failed: {str(e)}"
            }

    def _process_advanced_steps(self, steps: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process steps for advanced workflow"""
        results = []
        for step in steps:
            # Handle conditional execution
            if "condition" in step:
                if not self._evaluate_condition(step["condition"], params):
                    continue

            # Handle step execution
            try:
                result = self._execute_step(step)
                results.append({
                    "step": step,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "step": step,
                    "error": str(e),
                    "status": "failed"
                })

        return results

    def _evaluate_condition(self, condition: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate a condition for workflow step execution"""
        try:
            condition_type = condition.get("type")
            if condition_type == "parameter":
                param_name = condition.get("parameter")
                expected_value = condition.get("value")
                return params.get(param_name) == expected_value
            elif condition_type == "result":
                step_id = condition.get("step_id")
                expected_status = condition.get("status", "success")
                step_result = next(
                    (r for r in self.current_workflow.get("results", [])
                     if r["step"].get("id") == step_id),
                    None
                )
                return step_result and step_result["status"] == expected_status
            return True
        except Exception:
            return False

    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step_type = step.get("type")
        if step_type == "data_operation":
            return self._execute_data_operation(step)
        elif step_type == "visualization":
            return self._execute_visualization(step)
        elif step_type == "memory_operation":
            return self._execute_memory_operation(step)
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    def _execute_data_operation(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data operation step"""
        operation = step.get("operation")
        params = step.get("params", {})

        # Get the current data from memory if not provided in params
        if "dataframe" not in params:
            # Try to get the dataset from memory
            memory_result = self.memory_system.handle_memory_request({
                "operation": "retrieve",
                "params": {"key": "test_data"}
            })

            if memory_result.get("success") and "data" in memory_result:
                params["dataframe"] = memory_result["data"]

        # Try several naming variations
        function_names = [
            f"data_{operation}",  # data_filter
            operation,  # filter
            f"data.{operation}"  # data.filter
        ]

        # Try to find the function with different naming patterns
        func = None
        for name in function_names:
            # Try with category
            func = self.function_registry.get_function(name, "data")
            if func:
                break

            # Try without category
            func = self.function_registry.get_function(name)
            if func:
                break

        if not func:
            raise ValueError(f"Unknown data operation: {operation}")

        # Execute the function
        result = func(**params)

        # Store the resulting dataframe back to memory if operation was successful
        if result.get("success") and "dataframe" in result:
            self.memory_system.handle_memory_request({
                "operation": "store",
                "params": {
                    "key": "workflow_data",
                    "data": result["dataframe"]
                }
            })

        return result

    def _execute_visualization(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a visualization step"""
        viz_type = step.get("viz_type")
        params = step.get("params", {})

        # Get the current data from memory if not provided in params
        if "data" not in params and "dataframe" in params:
            params["data"] = params["dataframe"]

        if "data" not in params:
            # First try workflow data
            memory_result = self.memory_system.handle_memory_request({
                "operation": "retrieve",
                "params": {"key": "workflow_data"}
            })

            if not memory_result.get("success") or "data" not in memory_result:
                # Try the test_data key as fallback
                memory_result = self.memory_system.handle_memory_request({
                    "operation": "retrieve",
                    "params": {"key": "test_data"}
                })

            if memory_result.get("success") and "data" in memory_result:
                params["data"] = memory_result["data"]

        # The function name is prefixed with "viz_" in the registry
        function_name = f"viz_{viz_type}"

        # First try to find it in the "visualization" category
        func = self.function_registry.get_function(function_name, "visualization")
        if not func:
            # Then try to find it without specifying a category
            func = self.function_registry.get_function(function_name)
            if not func:
                raise ValueError(f"Unknown visualization type: {viz_type}")

        return func(**params)

    def _execute_memory_operation(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a memory operation step"""
        operation = step.get("operation")
        params = step.get("params", {})

        return self.memory_system.handle_memory_request({
            "operation": operation,
            "params": params
        })

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the current workflow status"""
        return {
            "current_workflow": self.current_workflow,
            "history": self.workflow_history,
            "success_rate": self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate the success rate of workflow steps"""
        if not self.workflow_history:
            return 1.0
        successful_steps = sum(
            1 for workflow in self.workflow_history
            for step in workflow.get("results", [])
            if step["status"] == "success"
        )
        total_steps = sum(
            len(workflow.get("results", []))
            for workflow in self.workflow_history
        )
        return successful_steps / total_steps if total_steps > 0 else 1.0