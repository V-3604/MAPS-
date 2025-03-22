# agent_config.py

from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path
import logging
from datetime import datetime
import json
from core.memory_system import MemorySystem


class AgentConfig:
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.logger = self._setup_logger()
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.default_configs: Dict[str, Dict[str, Any]] = self._load_default_configs()
        self.agent_states: Dict[str, str] = {}
        self.config_history: List[Dict[str, Any]] = []
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self._load_validation_rules()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        Path('output/logs').mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler('output/logs/agent_config.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load default configurations for different agent types"""
        return {
            "data_engineer": {
                "max_memory": "1GB",
                "timeout": 300,
                "batch_size": 1000,
                "processing_mode": "stream",
                "auto_optimize": True,
                "cache_enabled": True,
                "retry_attempts": 3,
                "logging_level": "INFO",
                "performance_monitoring": True,
                "error_handling": {
                    "mode": "strict",
                    "max_retries": 3,
                    "fallback_strategy": "skip"
                }
            },
            "viz_specialist": {
                "max_memory": "2GB",
                "timeout": 600,
                "default_style": "seaborn",
                "interactive": True,
                "max_fig_size": [12, 8],
                "dpi": 300,
                "save_format": "png",
                "color_scheme": "default",
                "animation_enabled": True,
                "rendering_backend": "auto",
                "cache_enabled": True
            },
            "memory_agent": {
                "max_history_size": 10000,
                "cleanup_interval": 86400,
                "compression_enabled": True,
                "indexing_enabled": True,
                "cache_size": "500MB",
                "sync_interval": 300,
                "backup_enabled": True,
                "recovery_mode": "auto",
                "persistence": {
                    "enabled": True,
                    "format": "pickle",
                    "interval": 3600
                }
            },
            "orchestrator": {
                "max_concurrent_agents": 10,
                "scheduling_algorithm": "round_robin",
                "load_balancing": True,
                "health_check_interval": 60,
                "timeout": 3600,
                "auto_scaling": {
                    "enabled": True,
                    "min_instances": 1,
                    "max_instances": 5,
                    "scale_up_threshold": 0.8,
                    "scale_down_threshold": 0.2
                }
            }
        }

    def _load_validation_rules(self):
        """Load validation rules for agent configurations"""
        self.validation_rules = {
            "data_engineer": {
                "max_memory": lambda x: isinstance(x, str) and x.endswith(('MB', 'GB')),
                "timeout": lambda x: isinstance(x, int) and 0 < x < 3600,
                "batch_size": lambda x: isinstance(x, int) and x > 0,
                "processing_mode": lambda x: x in ["stream", "batch"],
                "retry_attempts": lambda x: isinstance(x, int) and x >= 0
            },
            "viz_specialist": {
                "max_memory": lambda x: isinstance(x, str) and x.endswith(('MB', 'GB')),
                "timeout": lambda x: isinstance(x, int) and 0 < x < 3600,
                "dpi": lambda x: isinstance(x, int) and 0 < x <= 1200,
                "max_fig_size": lambda x: isinstance(x, list) and len(x) == 2
            },
            "memory_agent": {
                "max_history_size": lambda x: isinstance(x, int) and x > 0,
                "cleanup_interval": lambda x: isinstance(x, int) and x > 0,
                "cache_size": lambda x: isinstance(x, str) and x.endswith(('MB', 'GB'))
            },
            "orchestrator": {
                "max_concurrent_agents": lambda x: isinstance(x, int) and x > 0,
                "health_check_interval": lambda x: isinstance(x, int) and x > 0,
                "timeout": lambda x: isinstance(x, int) and x > 0
            }
        }

    def create_agent_config(self, agent_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent configuration"""
        try:
            if agent_type not in self.default_configs:
                return {"success": False, "error": f"Unknown agent type: {agent_type}"}

            # Merge with default config
            merged_config = self.default_configs[agent_type].copy()
            merged_config.update(config)

            # Validate configuration
            validation_result = self.validate_config(agent_type, merged_config)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Invalid configuration",
                    "validation_issues": validation_result["issues"]
                }

            # Store configuration
            self.agent_configs[agent_type] = merged_config
            self.agent_states[agent_type] = "configured"

            # Record change
            self._record_config_change(agent_type, "create", merged_config)

            self.logger.info(f"Created configuration for agent type: {agent_type}")
            return {
                "success": True,
                "message": f"Configuration created for {agent_type}",
                "config": merged_config
            }

        except Exception as e:
            self.logger.error(f"Error creating agent configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_agent_config(self, agent_type: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing agent configuration"""
        try:
            if agent_type not in self.agent_configs:
                return {"success": False, "error": f"No configuration exists for agent type: {agent_type}"}

            # Create updated config
            updated_config = self.agent_configs[agent_type].copy()
            updated_config.update(updates)

            # Validate updated configuration
            validation_result = self.validate_config(agent_type, updated_config)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Invalid configuration",
                    "validation_issues": validation_result["issues"]
                }

            # Store updated configuration
            self.agent_configs[agent_type] = updated_config
            self._record_config_change(agent_type, "update", updates)

            self.logger.info(f"Updated configuration for agent type: {agent_type}")
            return {
                "success": True,
                "message": f"Configuration updated for {agent_type}",
                "config": updated_config
            }

        except Exception as e:
            self.logger.error(f"Error updating agent configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent type"""
        try:
            if agent_type not in self.agent_configs:
                return {"success": False, "error": f"No configuration exists for agent type: {agent_type}"}

            return {
                "success": True,
                "config": self.agent_configs[agent_type],
                "state": self.agent_states.get(agent_type, "unknown")
            }

        except Exception as e:
            self.logger.error(f"Error retrieving agent configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def validate_config(self, agent_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent configuration against rules"""
        try:
            issues = []
            rules = self.validation_rules.get(agent_type, {})

            for key, validator in rules.items():
                if key in config:
                    if not validator(config[key]):
                        issues.append(f"Invalid value for {key}: {config[key]}")

            return {
                "valid": len(issues) == 0,
                "issues": issues
            }

        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return {"valid": False, "issues": [str(e)]}

    def _record_config_change(self, agent_type: str, action: str, details: Dict[str, Any]):
        """Record configuration change in history"""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "action": action,
            "details": details
        }
        self.config_history.append(change_record)

        # Store in memory system
        self.memory_system.add_operation(
            operation_type="config_change",
            details=change_record,
            result=None
        )

    def save_configs(self, path: str) -> Dict[str, Any]:
        """Save all agent configurations to file"""
        try:
            save_data = {
                "configurations": self.agent_configs,
                "states": self.agent_states,
                "history": self.config_history,
                "timestamp": datetime.now().isoformat()
            }

            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                json.dump(save_data, f, indent=2)

            self.logger.info(f"Saved configurations to {path}")
            return {"success": True, "message": f"Configurations saved to {path}"}

        except Exception as e:
            self.logger.error(f"Error saving configurations: {str(e)}")
            return {"success": False, "error": str(e)}

    def load_configs(self, path: str) -> Dict[str, Any]:
        """Load agent configurations from file"""
        try:
            with open(path, 'r') as f:
                load_data = json.load(f)

            self.agent_configs = load_data["configurations"]
            self.agent_states = load_data["states"]
            self.config_history = load_data["history"]

            self.logger.info(f"Loaded configurations from {path}")
            return {"success": True, "message": "Configurations loaded successfully"}

        except Exception as e:
            self.logger.error(f"Error loading configurations: {str(e)}")
            return {"success": False, "error": str(e)}

    def reset_config(self, agent_type: str) -> Dict[str, Any]:
        """Reset agent configuration to default"""
        try:
            if agent_type not in self.default_configs:
                return {"success": False, "error": f"Unknown agent type: {agent_type}"}

            self.agent_configs[agent_type] = self.default_configs[agent_type].copy()
            self.agent_states[agent_type] = "default"
            self._record_config_change(agent_type, "reset", {})

            self.logger.info(f"Reset configuration for agent type: {agent_type}")
            return {
                "success": True,
                "message": f"Configuration reset for {agent_type}",
                "config": self.agent_configs[agent_type]
            }

        except Exception as e:
            self.logger.error(f"Error resetting configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_config_history(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration change history"""
        try:
            if agent_type:
                history = [
                    change for change in self.config_history
                    if change["agent_type"] == agent_type
                ]
            else:
                history = self.config_history

            return {
                "success": True,
                "history": history
            }

        except Exception as e:
            self.logger.error(f"Error retrieving configuration history: {str(e)}")
            return {"success": False, "error": str(e)}