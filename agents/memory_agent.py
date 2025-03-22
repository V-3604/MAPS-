# memory_agent.py

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging
from core.memory_system import MemorySystem


class MemoryAgent:
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.context_history = []
        self.checkpoint_directory = Path("data/checkpoints")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.active_states = {}
        self.metadata_store = {}

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('output/logs/memory_agent.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def store_context(self, context: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store new context information with optional metadata"""
        try:
            timestamp = datetime.now().isoformat()
            context_entry = {
                "timestamp": timestamp,
                "context": context,
                "type": context.get("type", "general"),
                "metadata": metadata or {}
            }

            self.context_history.append(context_entry)
            self.metadata_store[timestamp] = metadata or {}

            self.memory_system.add_operation(
                operation_type="context_store",
                details=context,
                result={"timestamp": timestamp}
            )

            self.logger.info(f"Stored context of type {context.get('type', 'general')}")
            return {
                "success": True,
                "message": "Context stored successfully",
                "data": context_entry
            }
        except Exception as e:
            self.logger.error(f"Error storing context: {str(e)}")
            self.memory_system.log_exception(e)
            return {"success": False, "error": str(e)}

    def retrieve_context(self, query: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Retrieve context based on query parameters with optional metadata"""
        try:
            matched_contexts = []

            for entry in self.context_history:
                if self._matches_query(entry, query):
                    if include_metadata:
                        entry["stored_metadata"] = self.metadata_store.get(entry["timestamp"], {})
                    matched_contexts.append(entry)

            self.logger.info(f"Retrieved {len(matched_contexts)} matching contexts")
            return {
                "success": True,
                "message": f"Found {len(matched_contexts)} matching contexts",
                "data": matched_contexts
            }
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            self.memory_system.log_exception(e)
            return {"success": False, "error": str(e)}

    def create_checkpoint(self, name: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """Create a checkpoint of current state with metadata"""
        try:
            timestamp = datetime.now().isoformat()
            checkpoint_path = self.checkpoint_directory / f"{name}_{timestamp}.json"

            checkpoint_data = {
                "timestamp": timestamp,
                "name": name,
                "data": data,
                "metadata": metadata or {},
                "context_history": self.context_history[-10:],  # Last 10 contexts
                "active_states": self.active_states
            }

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            self.memory_system.add_operation(
                operation_type="checkpoint_create",
                details={"name": name, "metadata": metadata},
                result={"path": str(checkpoint_path)}
            )

            self.logger.info(f"Created checkpoint: {name}")
            return {
                "success": True,
                "message": "Checkpoint created successfully",
                "data": {"path": str(checkpoint_path)}
            }
        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {str(e)}")
            self.memory_system.log_exception(e)
            return {"success": False, "error": str(e)}

    def restore_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Restore state from a checkpoint"""
        try:
            checkpoint_files = list(self.checkpoint_directory.glob(f"{checkpoint_name}_*.json"))
            if not checkpoint_files:
                return {"success": False, "error": "Checkpoint not found"}

            # Get most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)

            self.context_history.extend(checkpoint_data["context_history"])
            self.active_states = checkpoint_data.get("active_states", {})

            self.memory_system.add_operation(
                operation_type="checkpoint_restore",
                details={"name": checkpoint_name},
                result={"timestamp": checkpoint_data["timestamp"]}
            )

            self.logger.info(f"Restored checkpoint: {checkpoint_name}")
            return {
                "success": True,
                "message": "Checkpoint restored successfully",
                "data": checkpoint_data
            }
        except Exception as e:
            self.logger.error(f"Error restoring checkpoint: {str(e)}")
            self.memory_system.log_exception(e)
            return {"success": False, "error": str(e)}

    def summarize_context(self, time_range: Optional[Dict[str, str]] = None,
                          include_metadata: bool = True) -> Dict[str, Any]:
        """Generate summary of context history with optional metadata"""
        try:
            contexts = self._filter_contexts_by_time(time_range) if time_range else self.context_history

            summary = {
                "total_contexts": len(contexts),
                "context_types": {},
                "time_range": {
                    "start": contexts[0]["timestamp"] if contexts else None,
                    "end": contexts[-1]["timestamp"] if contexts else None
                },
                "metadata_summary": {} if include_metadata else None
            }

            # Summarize context types and metadata
            for context in contexts:
                context_type = context["context"].get("type", "general")
                if context_type not in summary["context_types"]:
                    summary["context_types"][context_type] = 0
                summary["context_types"][context_type] += 1

                if include_metadata:
                    metadata = self.metadata_store.get(context["timestamp"], {})
                    for key, value in metadata.items():
                        if key not in summary["metadata_summary"]:
                            summary["metadata_summary"][key] = set()
                        summary["metadata_summary"][key].add(str(value))

            # Convert sets to lists for JSON serialization
            if include_metadata:
                summary["metadata_summary"] = {
                    k: list(v) for k, v in summary["metadata_summary"].items()
                }

            self.logger.info("Generated context summary")
            return {
                "success": True,
                "message": "Context summary generated successfully",
                "data": summary
            }
        except Exception as e:
            self.logger.error(f"Error generating context summary: {str(e)}")
            self.memory_system.log_exception(e)
            return {"success": False, "error": str(e)}

    def _matches_query(self, entry: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if context entry matches query parameters"""
        for key, value in query.items():
            if key == "time_range":
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if not (datetime.fromisoformat(value["start"]) <= entry_time <= datetime.fromisoformat(value["end"])):
                    return False
            elif key == "type":
                if entry["context"].get("type") != value:
                    return False
            elif key == "contains":
                if not any(value in str(v) for v in entry["context"].values()):
                    return False
            elif key == "metadata":
                entry_metadata = self.metadata_store.get(entry["timestamp"], {})
                if not all(entry_metadata.get(k) == v for k, v in value.items()):
                    return False
        return True

    def _filter_contexts_by_time(self, time_range: Dict[str, str]) -> List[Dict[str, Any]]:
        """Filter contexts by time range"""
        start_time = datetime.fromisoformat(time_range["start"])
        end_time = datetime.fromisoformat(time_range["end"])

        return [
            context for context in self.context_history
            if start_time <= datetime.fromisoformat(context["timestamp"]) <= end_time
        ]

    def cleanup_old_contexts(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up contexts older than specified days"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)

            original_count = len(self.context_history)
            original_metadata_count = len(self.metadata_store)

            self.context_history = [
                context for context in self.context_history
                if datetime.fromisoformat(context["timestamp"]).timestamp() > cutoff_date
            ]

            # Clean up metadata store
            self.metadata_store = {
                timestamp: metadata
                for timestamp, metadata in self.metadata_store.items()
                if datetime.fromisoformat(timestamp).timestamp() > cutoff_date
            }

            removed_contexts = original_count - len(self.context_history)
            removed_metadata = original_metadata_count - len(self.metadata_store)

            self.memory_system.add_operation(
                operation_type="context_cleanup",
                details={"days_kept": days_to_keep},
                result={
                    "removed_contexts": removed_contexts,
                    "removed_metadata": removed_metadata
                }
            )

            self.logger.info(f"Cleaned up {removed_contexts} contexts and {removed_metadata} metadata entries")
            return {
                "success": True,
                "message": f"Removed {removed_contexts} old contexts and {removed_metadata} metadata entries",
                "data": {
                    "removed_contexts": removed_contexts,
                    "removed_metadata": removed_metadata
                }
            }
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            self.memory_system.log_exception(e)
            return {"success": False, "error": str(e)}