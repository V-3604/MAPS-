# system_config.py

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import logging
from datetime import datetime
import json
import os
import shutil


class SystemConfig:
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.config = self._load_default_config()
        self.logger = self._setup_logger()
        self.history = []
        self.environment_overrides = {}
        self.load_config()
        self._setup_directories()
        self._apply_environment_variables()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        Path('output/logs').mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler('output/logs/system_config.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default system configuration"""
        return {
            "system": {
                "debug_mode": False,
                "log_level": "INFO",
                "max_memory_usage": "2GB",
                "timeout": 300,
                "auto_cleanup": True,
                "environment": "development",
                "version": "1.0.0",
                "encoding": "utf-8"
            },
            "memory": {
                "max_history_size": 1000,
                "checkpoint_interval": 60,
                "cleanup_interval": 86400,
                "max_checkpoint_age": 30,
                "compression_enabled": True,
                "compression_level": 6
            },
            "agents": {
                "max_concurrent": 5,
                "timeout": 120,
                "retry_attempts": 3,
                "retry_delay": 5,
                "health_check_interval": 60,
                "auto_scaling": True,
                "min_instances": 1,
                "max_instances": 10
            },
            "workflows": {
                "max_steps": 100,
                "max_execution_time": 3600,
                "auto_checkpoint": True,
                "checkpoint_frequency": 10,
                "parallel_execution": True,
                "max_parallel_branches": 4
            },
            "data": {
                "max_file_size": "1GB",
                "supported_formats": ["csv", "json", "xlsx", "parquet", "pickle"],
                "temp_directory": "data/temp",
                "cache_enabled": True,
                "cache_size": "500MB",
                "compression_format": "gzip",
                "backup_enabled": True,
                "backup_interval": 86400
            },
            "visualization": {
                "default_style": "seaborn",
                "max_fig_size": [12, 8],
                "dpi": 300,
                "interactive": True,
                "save_format": "png",
                "color_scheme": "default",
                "font_size": 10
            },
            "security": {
                "enable_logging": True,
                "max_retries": 3,
                "timeout": 30,
                "allowed_operations": ["read", "write", "execute"],
                "encryption_enabled": True,
                "encryption_algorithm": "AES-256",
                "ssl_verify": True
            },
            "paths": {
                "data": "data",
                "output": "output",
                "logs": "output/logs",
                "checkpoints": "data/checkpoints",
                "temp": "data/temp",
                "cache": "data/cache",
                "backup": "data/backup"
            },
            "optimization": {
                "cache_strategy": "lru",
                "prefetch_enabled": True,
                "batch_size": 1000,
                "thread_pool_size": 4,
                "process_pool_size": 2
            }
        }

    def _setup_directories(self):
        """Create necessary directories based on configuration"""
        try:
            for path_key, path_value in self.config["paths"].items():
                Path(path_value).mkdir(parents=True, exist_ok=True)
            self.logger.info("Directory structure setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up directories: {str(e)}")

    def _apply_environment_variables(self):
        """Apply environment variable overrides to configuration"""
        try:
            for key, value in os.environ.items():
                if key.startswith("SYSTEM_CONFIG_"):
                    config_key = key.replace("SYSTEM_CONFIG_", "").lower()
                    self.environment_overrides[config_key] = value
                    self._set_nested_config(config_key.split('_'), value)
            self.logger.info("Environment variables applied to configuration")
        except Exception as e:
            self.logger.error(f"Error applying environment variables: {str(e)}")

    def _set_nested_config(self, key_path: list, value: Any):
        """Set a nested configuration value"""
        current = self.config
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[key_path[-1]] = value

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                self._update_nested_dict(self.config, file_config)
                self._record_change("load_config", "Configuration loaded from file")
                self.logger.info("Configuration loaded successfully")
                return {"success": True, "message": "Configuration loaded successfully"}
            else:
                self.save_config()
                return {"success": True, "message": "Default configuration created"}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_config(self) -> Dict[str, Any]:
        """Save current configuration to file"""
        try:
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)

            # Create backup of existing config
            if Path(self.config_path).exists():
                backup_path = f"{self.config_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(self.config_path, backup_path)

            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

            self._record_change("save_config", "Configuration saved to file")
            self.logger.info("Configuration saved successfully")
            return {"success": True, "message": "Configuration saved successfully"}
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_config(self, updates: Dict[str, Any], section: Optional[str] = None) -> Dict[str, Any]:
        """Update configuration with new values"""
        try:
            if section:
                if section not in self.config:
                    return {"success": False, "error": f"Section {section} not found"}
                self._update_nested_dict(self.config[section], updates)
            else:
                self._update_nested_dict(self.config, updates)

            self._record_change("update_config", f"Configuration updated: {updates}")
            self.logger.info(f"Configuration updated: {section if section else 'root'}")
            return {"success": True, "message": "Configuration updated successfully"}
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Update nested dictionary with new values"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v

    def get_config(self, section: Optional[str] = None, key: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration values"""
        try:
            if section and key:
                if section not in self.config or key not in self.config[section]:
                    return {"success": False, "error": f"Config path {section}.{key} not found"}
                return {"success": True, "data": self.config[section][key]}
            elif section:
                if section not in self.config:
                    return {"success": False, "error": f"Section {section} not found"}
                return {"success": True, "data": self.config[section]}
            else:
                return {"success": True, "data": self.config}
        except Exception as e:
            self.logger.error(f"Error getting configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def _record_change(self, action: str, description: str):
        """Record configuration change in history"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "description": description
        })

    def get_history(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get configuration change history"""
        try:
            history = self.history[-limit:] if limit else self.history
            return {"success": True, "history": history}
        except Exception as e:
            self.logger.error(f"Error getting history: {str(e)}")
            return {"success": False, "error": str(e)}

    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        try:
            issues = []

            # Validate system settings
            if not isinstance(self.config.get("system", {}).get("timeout"), (int, float)):
                issues.append("Invalid system timeout value")

            # Validate memory settings
            memory_config = self.config.get("memory", {})
            if memory_config.get("max_history_size", 0) <= 0:
                issues.append("Invalid max_history_size")

            # Validate paths
            for path_key, path_value in self.config.get("paths", {}).items():
                if not isinstance(path_value, str):
                    issues.append(f"Invalid path value for {path_key}")

            return {
                "valid": len(issues) == 0,
                "issues": issues
            }
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return {"valid": False, "issues": [str(e)]}

    def reset_to_default(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Reset configuration to default values"""
        try:
            default_config = self._load_default_config()
            if section:
                if section not in self.config:
                    return {"success": False, "error": f"Section {section} not found"}
                self.config[section] = default_config[section]
            else:
                self.config = default_config

            self._record_change("reset_to_default", f"Reset to default: {section if section else 'all'}")
            self.logger.info(f"Configuration reset to default: {section if section else 'all'}")
            return {"success": True, "message": "Configuration reset to default successfully"}
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def export_config(self, export_path: str, format: str = "yaml") -> Dict[str, Any]:
        """Export configuration to file in specified format"""
        try:
            export_dir = Path(export_path).parent
            export_dir.mkdir(parents=True, exist_ok=True)

            if format.lower() == "yaml":
                with open(export_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif format.lower() == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                return {"success": False, "error": f"Unsupported export format: {format}"}

            self.logger.info(f"Configuration exported to {export_path}")
            return {"success": True, "message": f"Configuration exported to {export_path}"}
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {str(e)}")
            return {"success": False, "error": str(e)}