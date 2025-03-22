# advanced_workflow.py

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
from pathlib import Path
import networkx as nx
from core.memory_system import MemorySystem
from core.function_registry import FunctionRegistry
from basic_workflow import BasicWorkflow
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, Future


class AdvancedWorkflow(BasicWorkflow):
    def __init__(self, memory_system: MemorySystem, function_registry: FunctionRegistry):
        super().__init__(memory_system, function_registry)
        self.workflow_graph = nx.DiGraph()
        self.conditional_functions: Dict[str, Callable] = {}
        self.parallel_executors: Dict[str, ThreadPoolExecutor] = {}
        self.futures: Dict[str, List[Future]] = {}
        self.logger = self._setup_logger()
        self.recovery_points = {}
        self.workflow_hooks = {
            "pre_execution": [],
            "post_execution": [],
            "error": []
        }

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('output/logs/advanced_workflow.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def create_workflow(self, steps: List[Dict[str, Any]], dependencies: Dict[str, List[str]] = None,
                        conditions: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an advanced workflow with dependencies and conditions"""
        try:
            workflow_id = str(uuid.uuid4())
            self.workflow_graph.clear()

            # Add nodes (steps)
            for i, step in enumerate(steps):
                step_id = str(i)
                self.workflow_graph.add_node(step_id, step_data=step)

            # Add edges (dependencies)
            if dependencies:
                for step_id, deps in dependencies.items():
                    for dep in deps:
                        self.workflow_graph.add_edge(dep, step_id)

            # Add conditions
            if conditions:
                for step_id, condition in conditions.items():
                    if step_id in self.workflow_graph.nodes:
                        self.workflow_graph.nodes[step_id]["condition"] = condition

            # Validate workflow
            validation = self._validate_workflow_graph()
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": "Invalid workflow graph",
                    "validation_issues": validation["issues"]
                }

            # Store workflow state
            self.workflow_states[workflow_id] = {
                "graph": self.workflow_graph.copy(),
                "status": "created",
                "creation_time": datetime.now().isoformat(),
                "executed_steps": set(),
                "results": {},
                "metadata": {}
            }

            self.logger.info(f"Created advanced workflow: {workflow_id}")
            return {
                "success": True,
                "workflow_id": workflow_id,
                "message": "Advanced workflow created successfully"
            }

        except Exception as e:
            self.logger.error(f"Error creating advanced workflow: {str(e)}")
            return {"success": False, "error": str(e)}

    def execute_workflow(self, workflow_id: str, parallel: bool = False,
                         max_workers: int = 4) -> Dict[str, Any]:
        """Execute the advanced workflow"""
        try:
            workflow_state = self.workflow_states.get(workflow_id)
            if not workflow_state:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}

            self._execute_hooks("pre_execution", workflow_id)

            workflow_state["status"] = "running"
            workflow_state["start_time"] = datetime.now().isoformat()

            if parallel:
                result = self._execute_parallel(workflow_id, max_workers)
            else:
                result = self._execute_sequential(workflow_id)

            workflow_state["end_time"] = datetime.now().isoformat()
            workflow_state["status"] = "completed" if result["success"] else "failed"

            self._execute_hooks("post_execution", workflow_id)

            self._save_workflow_results(workflow_state)
            return result

        except Exception as e:
            self.logger.error(f"Error executing advanced workflow: {str(e)}")
            self._execute_hooks("error", workflow_id, error=e)
            return {"success": False, "error": str(e)}

    def _execute_sequential(self, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow steps sequentially"""
        workflow_state = self.workflow_states[workflow_id]
        graph = workflow_state["graph"]

        try:
            execution_order = list(nx.topological_sort(graph))
            results = {}

            for step_id in execution_order:
                step_data = graph.nodes[step_id]["step_data"]

                # Check conditions
                if "condition" in graph.nodes[step_id]:
                    if not self._evaluate_condition(graph.nodes[step_id]["condition"], results):
                        continue

                # Check dependencies
                deps_results = {
                    dep: results[dep]
                    for dep in graph.predecessors(step_id)
                    if dep in results
                }

                if any(not r["success"] for r in deps_results.values()):
                    results[step_id] = {"success": False, "error": "Dependency execution failed"}
                    if step_data.get("critical", False):
                        break
                    continue

                # Execute step
                step_result = self._execute_advanced_step(step_data, deps_results)
                results[step_id] = step_result
                workflow_state["executed_steps"].add(step_id)
                workflow_state["results"][step_id] = step_result

                if not step_result["success"] and step_data.get("critical", False):
                    break

            workflow_state["results"] = results
            return {
                "success": all(r["success"] for r in results.values()),
                "results": results
            }

        except Exception as e:
            self.logger.error(f"Error in sequential execution: {str(e)}")
            return {"success": False, "error": str(e)}

    def _execute_parallel(self, workflow_id: str, max_workers: int) -> Dict[str, Any]:
        """Execute workflow steps in parallel where possible"""
        workflow_state = self.workflow_states[workflow_id]
        graph = workflow_state["graph"]

        try:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            self.parallel_executors[workflow_id] = executor
            self.futures[workflow_id] = []
            results = {}

            # Find all independent paths for parallel execution
            paths = self._find_parallel_paths(graph)

            for path in paths:
                future = executor.submit(self._execute_path, workflow_id, path)
                self.futures[workflow_id].append(future)

            # Wait for all paths to complete
            for future in self.futures[workflow_id]:
                path_results = future.result()
                results.update(path_results)

            workflow_state["results"] = results
            executor.shutdown()

            return {
                "success": all(r["success"] for r in results.values()),
                "results": results
            }

        except Exception as e:
            self.logger.error(f"Error in parallel execution: {str(e)}")
            return {"success": False, "error": str(e)}

    def _execute_path(self, workflow_id: str, path: List[str]) -> Dict[str, Dict[str, Any]]:
        """Execute a single path of the workflow"""
        workflow_state = self.workflow_states[workflow_id]
        graph = workflow_state["graph"]
        results = {}

        for step_id in path:
            step_data = graph.nodes[step_id]["step_data"]

            # Check conditions and dependencies
            if not self._can_execute_step(step_id, graph, results):
                continue

            deps_results = {
                dep: results[dep]
                for dep in graph.predecessors(step_id)
                if dep in results
            }

            step_result = self._execute_advanced_step(step_data, deps_results)
            results[step_id] = step_result

            with self._workflow_state_lock:
                workflow_state["executed_steps"].add(step_id)
                workflow_state["results"][step_id] = step_result

            if not step_result["success"] and step_data.get("critical", False):
                break

        return results

    def _find_parallel_paths(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find independent paths that can be executed in parallel"""
        paths = []
        visited = set()

        def dfs(node: str, current_path: List[str]):
            if node not in visited:
                visited.add(node)
                current_path.append(node)

                successors = list(graph.successors(node))
                if not successors:
                    paths.append(current_path[:])
                else:
                    for successor in successors:
                        dfs(successor, current_path)

                current_path.pop()

        start_nodes = [n for n in graph.nodes() if not list(graph.predecessors(n))]
        for start_node in start_nodes:
            dfs(start_node, [])

        return paths

    def _execute_advanced_step(self, step_data: Dict[str, Any],
                               deps_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single step with advanced features"""
        try:
            # Update parameters with dependency results if specified
            params = step_data.get("params", {}).copy()
            if "param_mapping" in step_data:
                for param, dep_info in step_data["param_mapping"].items():
                    dep_id = dep_info["step_id"]
                    dep_key = dep_info["result_key"]
                    if dep_id in deps_results and deps_results[dep_id]["success"]:
                        params[param] = deps_results[dep_id]["result"][dep_key]

            # Execute step with updated parameters
            return super()._execute_step({
                "type": step_data["type"],
                "function": step_data["function"],
                "params": params
            })

        except Exception as e:
            self.logger.error(f"Error executing advanced step: {str(e)}")
            return {"success": False, "error": str(e)}

    def _evaluate_condition(self, condition: Dict[str, Any],
                            results: Dict[str, Dict[str, Any]]) -> bool:
        """Evaluate a condition for conditional execution"""
        try:
            condition_type = condition["type"]

            if condition_type == "step_success":
                step_id = condition["step_id"]
                return results.get(step_id, {}).get("success", False)

            elif condition_type == "custom":
                func_name = condition["function"]
                if func_name not in self.conditional_functions:
                    raise ValueError(f"Conditional function not found: {func_name}")

                return self.conditional_functions[func_name](
                    condition.get("params", {}),
                    results
                )

            elif condition_type == "expression":
                # Evaluate Python expression with results context
                return eval(condition["expression"], {"results": results})

            else:
                raise ValueError(f"Unknown condition type: {condition_type}")

        except Exception as e:
            self.logger.error(f"Error evaluating condition: {str(e)}")
            return False

    def _validate_workflow_graph(self) -> Dict[str, Any]:
        """Validate the workflow graph"""
        try:
            issues = []

            # Check for cycles
            if not nx.is_directed_acyclic_graph(self.workflow_graph):
                issues.append("Workflow contains cycles")

            # Validate each step
            for node_id in self.workflow_graph.nodes:
                step_data = self.workflow_graph.nodes[node_id]["step_data"]

                # Validate step data
                if "type" not in step_data:
                    issues.append(f"Step {node_id}: Missing 'type' field")
                if "function" not in step_data:
                    issues.append(f"Step {node_id}: Missing 'function' field")

                # Validate conditions
                if "condition" in self.workflow_graph.nodes[node_id]:
                    condition = self.workflow_graph.nodes[node_id]["condition"]
                    if "type" not in condition:
                        issues.append(f"Step {node_id}: Invalid condition format")
                    elif condition["type"] == "custom" and condition.get("function") not in self.conditional_functions:
                        issues.append(f"Step {node_id}: Unknown conditional function")

                # Validate parameter mappings
                if "param_mapping" in step_data:
                    for param, mapping in step_data["param_mapping"].items():
                        if "step_id" not in mapping or "result_key" not in mapping:
                            issues.append(f"Step {node_id}: Invalid parameter mapping format")

            return {
                "valid": len(issues) == 0,
                "issues": issues
            }

        except Exception as e:
            self.logger.error(f"Error validating workflow graph: {str(e)}")
            return {"valid": False, "issues": [str(e)]}

    def register_conditional_function(self, name: str, func: Callable) -> Dict[str, Any]:
        """Register a custom conditional function"""
        try:
            self.conditional_functions[name] = func
            return {"success": True, "message": f"Conditional function {name} registered"}
        except Exception as e:
            self.logger.error(f"Error registering conditional function: {str(e)}")
            return {"success": False, "error": str(e)}

    def register_hook(self, hook_type: str, hook_function: Callable) -> Dict[str, Any]:
        """Register a workflow hook"""
        try:
            if hook_type not in self.workflow_hooks:
                return {"success": False, "error": f"Invalid hook type: {hook_type}"}

            self.workflow_hooks[hook_type].append(hook_function)
            return {"success": True, "message": f"Hook registered for {hook_type}"}
        except Exception as e:
            self.logger.error(f"Error registering hook: {str(e)}")
            return {"success": False, "error": str(e)}

    def _execute_hooks(self, hook_type: str, workflow_id: str, **kwargs):
        """Execute all registered hooks of a specific type"""
        try:
            for hook in self.workflow_hooks[hook_type]:
                hook(workflow_id, self.workflow_states[workflow_id], **kwargs)
        except Exception as e:
            self.logger.error(f"Error executing {hook_type} hooks: {str(e)}")

    def create_recovery_point(self, workflow_id: str) -> Dict[str, Any]:
        """Create a recovery point for the workflow"""
        try:
            workflow_state = self.workflow_states.get(workflow_id)
            if not workflow_state:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}

            recovery_point = {
                "timestamp": datetime.now().isoformat(),
                "executed_steps": workflow_state["executed_steps"].copy(),
                "results": workflow_state["results"].copy(),
                "status": workflow_state["status"]
            }

            self.recovery_points[workflow_id] = recovery_point
            return {"success": True, "message": "Recovery point created successfully"}
        except Exception as e:
            self.logger.error(f"Error creating recovery point: {str(e)}")
            return {"success": False, "error": str(e)}

    def restore_from_recovery(self, workflow_id: str) -> Dict[str, Any]:
        """Restore workflow from recovery point"""
        try:
            recovery_point = self.recovery_points.get(workflow_id)
            if not recovery_point:
                return {"success": False, "error": f"No recovery point found for workflow {workflow_id}"}

            workflow_state = self.workflow_states.get(workflow_id)
            if not workflow_state:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}

            workflow_state["executed_steps"] = recovery_point["executed_steps"]
            workflow_state["results"] = recovery_point["results"]
            workflow_state["status"] = recovery_point["status"]

            return {"success": True, "message": "Workflow restored from recovery point"}
        except Exception as e:
            self.logger.error(f"Error restoring from recovery point: {str(e)}")
            return {"success": False, "error": str(e)}