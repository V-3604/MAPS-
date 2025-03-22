# core/function_registry.py
from typing import Dict, Any, Callable, Optional


class FunctionRegistry:
    """Registry for system functions that can be called by components"""

    def __init__(self):
        self.functions = {}

    def register_function(self, func: Callable, category: Optional[str] = None, name: Optional[str] = None) -> bool:
        """
        Register a function with the registry

        Parameters:
        -----------
        func : Callable
            The function to register
        category : str, optional
            Category to group related functions
        name : str, optional
            Custom name for the function, defaults to function name

        Returns:
        --------
        bool
            Registration success
        """
        try:
            func_name = name or func.__name__

            # Create category if it doesn't exist
            if category:
                if category not in self.functions:
                    self.functions[category] = {}
                self.functions[category][func_name] = func
            else:
                # Store in root
                self.functions[func_name] = func

            return True
        except Exception:
            return False

    def get_function(self, name: str, category: Optional[str] = None) -> Optional[Callable]:
        """
        Get a registered function

        Parameters:
        -----------
        name : str
            Function name
        category : str, optional
            Category to look in

        Returns:
        --------
        Callable or None
            The registered function or None if not found
        """
        try:
            if category:
                if category in self.functions and name in self.functions[category]:
                    return self.functions[category][name]

                # Try alternative syntaxes if not found
                if "." in name:
                    parts = name.split(".")
                    if len(parts) == 2 and parts[0] == category:
                        if parts[1] in self.functions[category]:
                            return self.functions[category][parts[1]]
            else:
                # Look directly in root functions
                if name in self.functions:
                    return self.functions[name]

                # Look across all categories
                for cat, funcs in self.functions.items():
                    if isinstance(funcs, dict) and name in funcs:
                        return funcs[name]

                    # Check for alternative syntax
                    if "." in name:
                        parts = name.split(".")
                        if len(parts) == 2 and parts[0] == cat and parts[1] in funcs:
                            return funcs[parts[1]]

            return None
        except Exception:
            return None

    def list_functions(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        List registered functions

        Parameters:
        -----------
        category : str, optional
            Category to list functions from

        Returns:
        --------
        Dict[str, Any]
            Dictionary of function information
        """
        try:
            if category:
                if category not in self.functions:
                    return {"success": False, "error": f"Category {category} not found"}

                # Get functions in this category
                funcs = self.functions[category]
                if not isinstance(funcs, dict):
                    return {"success": False, "error": f"Category {category} is not a valid category"}

                return {
                    "success": True,
                    "category": category,
                    "functions": list(funcs.keys())
                }
            else:
                # List all functions by category
                result = {"success": True, "categories": {}}

                for key, value in self.functions.items():
                    if isinstance(value, dict):
                        # This is a category
                        result["categories"][key] = list(value.keys())
                    else:
                        # This is a root function
                        if "root" not in result:
                            result["root"] = []
                        result["root"].append(key)

                return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update registry configuration

        Parameters:
        -----------
        config_updates : Dict[str, Any]
            Configuration updates

        Returns:
        --------
        Dict[str, Any]
            Update result
        """
        # This is a placeholder for actual configuration logic
        return {"success": True, "message": "Function registry configuration updated"}