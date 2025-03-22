# core/memory_system.py

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle
import shutil
from collections import deque
import threading
from queue import Queue
import hashlib
import time
import os
from concurrent.futures import ThreadPoolExecutor
import traceback


class MemorySystem:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.operation_history = deque(maxlen=self.config["max_history_size"])
        self.data_store = {}
        self.metadata_store = {}
        self.checkpoints = {}
        self.context_store = {}
        self.index_store = {}
        self.cache = {}
        self._running = True
        self.logger = self._setup_logger()
        self.operation_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
        self._setup_directories()
        self.locks = {
            "data": threading.Lock(),
            "metadata": threading.Lock(),
            "operation": threading.Lock(),
            "backup": threading.Lock(),
            "cache": threading.Lock(),
            "index": threading.Lock()
        }
        self._start_background_tasks()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "max_history_size": 1000,
            "checkpoint_interval": 3600,  # seconds
            "cleanup_interval": 86400,  # seconds
            "max_checkpoint_age": 30,  # days
            "compression_enabled": True,
            "backup_enabled": True,
            "backup_interval": 86400,  # seconds
            "cache_size": "500MB",
            "index_enabled": True,
            "async_operations": True,
            "auto_cleanup": True,
            "max_retries": 3,
            "retry_delay": 5,
            "backup_retention": 5,
            "temp_cleanup_age": 24,
            "max_workers": 4,
            "cache_ttl": 3600,
            "index_update_batch_size": 100,
            "compression_level": 6,
            "max_cache_items": 1000
        }

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        Path('output/logs').mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler('output/logs/memory_system.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _setup_directories(self):
        """Create necessary directories for memory system"""
        directories = [
            "data/memory",
            "data/checkpoints",
            "data/backups",
            "data/indexes",
            "data/cache",
            "data/temp",
            "data/archives"
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            if self.config["async_operations"]:
                self.operation_thread = threading.Thread(
                    target=self._process_operation_queue,
                    daemon=True,
                    name="OperationQueue"
                )
                self.operation_thread.start()

            if self.config["backup_enabled"]:
                self.backup_thread = threading.Thread(
                    target=self._scheduled_backup,
                    daemon=True,
                    name="BackupTask"
                )
                self.backup_thread.start()

            if self.config["auto_cleanup"]:
                self.cleanup_thread = threading.Thread(
                    target=self._scheduled_cleanup,
                    daemon=True,
                    name="CleanupTask"
                )
                self.cleanup_thread.start()

            # Start cache maintenance thread
            self.cache_thread = threading.Thread(
                target=self._maintain_cache,
                daemon=True,
                name="CacheMaintenance"
            )
            self.cache_thread.start()

            # Start index maintenance thread
            if self.config["index_enabled"]:
                self.index_thread = threading.Thread(
                    target=self._maintain_index,
                    daemon=True,
                    name="IndexMaintenance"
                )
                self.index_thread.start()

            self.logger.info("Background tasks started successfully")
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {str(e)}")
            raise

    def _scheduled_backup(self):
        """Background task for scheduled backups"""
        while self._running:
            try:
                if self.config["backup_enabled"]:
                    with self.locks["backup"]:
                        self.create_backup()
                        self._cleanup_old_backups()
                    self.logger.info("Scheduled backup completed")

                time.sleep(self.config["backup_interval"])

            except Exception as e:
                self.logger.error(f"Error in scheduled backup: {str(e)}")
                time.sleep(300)

    def _scheduled_cleanup(self):
        """Background task for scheduled cleanup"""
        while self._running:
            try:
                if self.config["auto_cleanup"]:
                    self.cleanup()
                    self.logger.info("Scheduled cleanup completed")

                time.sleep(self.config["cleanup_interval"])

            except Exception as e:
                self.logger.error(f"Error in scheduled cleanup: {str(e)}")
                time.sleep(300)

    def _maintain_cache(self):
        """Background task for cache maintenance"""
        while self._running:
            try:
                with self.locks["cache"]:
                    current_time = time.time()
                    # Remove expired cache entries
                    expired_keys = [
                        k for k, v in self.cache.items()
                        if current_time - v["timestamp"] > self.config["cache_ttl"]
                    ]
                    for key in expired_keys:
                        del self.cache[key]

                    # Trim cache if it exceeds max size
                    if len(self.cache) > self.config["max_cache_items"]:
                        sorted_cache = sorted(
                            self.cache.items(),
                            key=lambda x: x[1]["timestamp"]
                        )
                        for key, _ in sorted_cache[:-self.config["max_cache_items"]]:
                            del self.cache[key]

                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in cache maintenance: {str(e)}")
                time.sleep(30)

    def _maintain_index(self):
        """Background task for index maintenance"""
        while self._running:
            try:
                with self.locks["index"]:
                    # Rebuild index if necessary
                    if len(self.data_store) != len(self.index_store):
                        self._rebuild_index()

                    # Verify index integrity
                    for key in list(self.index_store.keys()):
                        if key not in self.data_store:
                            del self.index_store[key]

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in index maintenance: {str(e)}")
                time.sleep(60)

    def _process_operation_queue(self):
        """Process operations from the queue"""
        while self._running:
            try:
                # Use a timeout to allow thread to check _running periodically
                try:
                    operation = self.operation_queue.get(timeout=1)
                    self._execute_operation(operation)
                    self.operation_queue.task_done()
                except Queue.Empty:
                    continue
            except Exception as e:
                self.logger.error(f"Error processing operation: {str(e)}")
                time.sleep(1)  # Sleep to avoid tight loop on error

    def _execute_operation(self, operation: Dict[str, Any]):
        """Execute a single operation"""
        try:
            operation_type = operation.get("type")
            if operation_type == "store":
                self._store_data_internal(
                    operation["key"],
                    operation["data"],
                    operation.get("metadata")
                )
            elif operation_type == "delete":
                self._delete_data_internal(operation["key"])
            elif operation_type == "update":
                self._update_data_internal(
                    operation["key"],
                    operation["data"],
                    operation.get("metadata")
                )
            elif operation_type == "batch":
                self._execute_batch_operation(operation["operations"])

            self._add_to_operation_history(
                operation_type,
                operation.get("key"),
                operation.get("metadata")
            )
        except Exception as e:
            self.logger.error(f"Error executing operation: {str(e)}")
            raise

    def handle_memory_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory operation requests"""
        try:
            operation = request.get("operation")
            params = request.get("params", {})

            if operation == "store":
                return self.store_data(params.get("key"), params.get("data"), params.get("metadata"))
            elif operation == "retrieve":
                return self.retrieve_data(params.get("key"), params.get("include_metadata", False))
            elif operation == "update":
                return self.update_data(params.get("key"), params.get("data"), params.get("metadata"))
            elif operation == "delete":
                return self.delete_data(params.get("key"))
            elif operation == "query":
                return self.query_data(params)
            elif operation == "checkpoint":
                return self.create_checkpoint(
                    params.get("name", f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
            elif operation == "restore":
                return self.restore_checkpoint(params.get("name"))
            elif operation == "stats":
                return self.get_stats()
            elif operation == "backup":
                return self.create_backup()
            elif operation == "cleanup":
                return self.cleanup()
            else:
                return {"success": False, "error": f"Unknown memory operation: {operation}"}
        except Exception as e:
            self.logger.error(f"Error handling memory request: {str(e)}")
            return {"success": False, "error": str(e)}

    def store_data(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store data with metadata"""
        try:
            if not key:
                key = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if self.config["async_operations"]:
                self.operation_queue.put({
                    "type": "store",
                    "key": key,
                    "data": data,
                    "metadata": metadata
                })
                return {"success": True, "message": "Storage operation queued", "key": key}
            else:
                result = self._store_data_internal(key, data, metadata)
                result["key"] = key
                return result

        except Exception as e:
            self.logger.error(f"Error storing data: {str(e)}")
            return {"success": False, "error": str(e)}

    def _store_data_internal(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal method for storing data"""
        try:
            with self.locks["data"]:
                if self.config["compression_enabled"] and isinstance(data, (dict, list, str)):
                    data = self._compress_data(data)
                self.data_store[key] = data

            if metadata:
                with self.locks["metadata"]:
                    self.metadata_store[key] = {
                        **metadata,
                        "timestamp": datetime.now().isoformat(),
                        "size": self._calculate_size(data),
                        "hash": self._calculate_hash(data)
                    }

            if self.config["index_enabled"]:
                self._update_index(key, data)

            # Update cache
            with self.locks["cache"]:
                self.cache[key] = {
                    "data": data,
                    "timestamp": time.time()
                }

            return {"success": True, "message": "Data stored successfully"}

        except Exception as e:
            self.logger.error(f"Error in internal data storage: {str(e)}")
            return {"success": False, "error": str(e)}

    def retrieve_data(self, key: str, include_metadata: bool = False) -> Dict[str, Any]:
        """Retrieve stored data"""
        try:
            # Check cache first
            with self.locks["cache"]:
                if key in self.cache:
                    cache_entry = self.cache[key]
                    if time.time() - cache_entry["timestamp"] <= self.config["cache_ttl"]:
                        data = cache_entry["data"]
                        self.logger.debug(f"Cache hit for key: {key}")
                    else:
                        del self.cache[key]
                        data = None
                else:
                    data = None

            if data is None:
                if key not in self.data_store:
                    return {"success": False, "error": "Key not found"}

                with self.locks["data"]:
                    data = self.data_store[key]
                    if self.config["compression_enabled"]:
                        data = self._decompress_data(data)

                # Update cache
                with self.locks["cache"]:
                    self.cache[key] = {
                        "data": data,
                        "timestamp": time.time()
                    }

            result = {
                "success": True,
                "data": data
            }

            if include_metadata:
                with self.locks["metadata"]:
                    if key in self.metadata_store:
                        result["metadata"] = self.metadata_store[key]

            self._add_to_operation_history("retrieve", key)
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_data(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update stored data and metadata"""
        try:
            if self.config["async_operations"]:
                self.operation_queue.put({
                    "type": "update",
                    "key": key,
                    "data": data,
                    "metadata": metadata
                })
                return {"success": True, "message": "Update operation queued"}
            else:
                return self._update_data_internal(key, data, metadata)

        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
            return {"success": False, "error": str(e)}

    def _update_data_internal(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal method for updating data"""
        try:
            if key not in self.data_store:
                return {"success": False, "error": "Key not found"}

            with self.locks["data"]:
                if self.config["compression_enabled"] and isinstance(data, (dict, list, str)):
                    data = self._compress_data(data)
                self.data_store[key] = data

            if metadata or key in self.metadata_store:
                with self.locks["metadata"]:
                    current_metadata = self.metadata_store.get(key, {})
                    self.metadata_store[key] = {
                        **current_metadata,
                        **(metadata or {}),
                        "updated_at": datetime.now().isoformat(),
                        "size": self._calculate_size(data),
                        "hash": self._calculate_hash(data)
                    }

            if self.config["index_enabled"]:
                self._update_index(key, data)

            # Update cache
            with self.locks["cache"]:
                self.cache[key] = {
                    "data": data,
                    "timestamp": time.time()
                }

            self._add_to_operation_history("update", key, metadata)
            return {"success": True, "message": "Data updated successfully"}

        except Exception as e:
            self.logger.error(f"Error in internal data update: {str(e)}")
            return {"success": False, "error": str(e)}

    def delete_data(self, key: str) -> Dict[str, Any]:
        """Delete stored data and metadata"""
        try:
            if self.config["async_operations"]:
                self.operation_queue.put({
                    "type": "delete",
                    "key": key
                })
                return {"success": True, "message": "Delete operation queued"}
            else:
                return self._delete_data_internal(key)

        except Exception as e:
            self.logger.error(f"Error deleting data: {str(e)}")
            return {"success": False, "error": str(e)}

    def _delete_data_internal(self, key: str) -> Dict[str, Any]:
        """Internal method for deleting data"""
        try:
            if key not in self.data_store:
                return {"success": False, "error": "Key not found"}

            with self.locks["data"]:
                del self.data_store[key]

            with self.locks["metadata"]:
                if key in self.metadata_store:
                    del self.metadata_store[key]

            if self.config["index_enabled"]:
                with self.locks["index"]:
                    if key in self.index_store:
                        del self.index_store[key]

            # Remove from cache
            with self.locks["cache"]:
                if key in self.cache:
                    del self.cache[key]

            self._add_to_operation_history("delete", key)
            return {"success": True, "message": "Data deleted successfully"}

        except Exception as e:
            self.logger.error(f"Error in internal data deletion: {str(e)}")
            return {"success": False, "error": str(e)}

    def query_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Query data based on metadata or content"""
        try:
            results = []
            query_type = query.get("type", "metadata")

            if query_type == "metadata":
                results = self._query_by_metadata(query.get("criteria", {}))
            elif query_type == "content":
                results = self._query_by_content(query.get("criteria", {}))
            elif query_type == "combined":
                metadata_results = self._query_by_metadata(query.get("metadata_criteria", {}))
                content_results = self._query_by_content(query.get("content_criteria", {}))
                results = list(set(metadata_results) & set(content_results))
            elif query_type == "index":
                results = self._query_by_index(query.get("criteria", {}))

            return {
                "success": True,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            self.logger.error(f"Error querying data: {str(e)}")
            return {"success": False, "error": str(e)}

    def _query_by_metadata(self, criteria: Dict[str, Any]) -> List[str]:
        """Query data by metadata criteria"""
        matching_keys = []
        with self.locks["metadata"]:
            for key, metadata in self.metadata_store.items():
                if all(
                        k in metadata and self._match_criterion(metadata[k], v)
                        for k, v in criteria.items()
                ):
                    matching_keys.append(key)
        return matching_keys

    def _query_by_content(self, criteria: Dict[str, Any]) -> List[str]:
        """Query data by content criteria"""
        matching_keys = []
        with self.locks["data"]:
            for key, data in self.data_store.items():
                if self._matches_content_criteria(data, criteria):
                    matching_keys.append(key)
        return matching_keys

    def _query_by_index(self, criteria: Dict[str, Any]) -> List[str]:
        """Query data using the index"""
        matching_keys = []
        with self.locks["index"]:
            for key, index_entry in self.index_store.items():
                if all(
                        k in index_entry and self._match_criterion(index_entry[k], v)
                        for k, v in criteria.items()
                ):
                    matching_keys.append(key)
        return matching_keys

    def _match_criterion(self, value: Any, criterion: Any) -> bool:
        """Match a single criterion against a value"""
        if isinstance(criterion, dict):
            op = criterion.get("op", "eq")
            target = criterion.get("value")

            if op == "eq":
                return value == target
            elif op == "ne":
                return value != target
            elif op == "gt":
                return value > target
            elif op == "gte":
                return value >= target
            elif op == "lt":
                return value < target
            elif op == "lte":
                return value <= target
            elif op == "in":
                return value in target
            elif op == "contains":
                return target in value
            else:
                return False
        else:
            return value == criterion

    def _matches_content_criteria(self, data: Any, criteria: Dict[str, Any]) -> bool:
        """Check if data matches content criteria"""
        if isinstance(data, dict):
            return all(
                k in data and self._match_criterion(data[k], v)
                for k, v in criteria.items()
            )
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return all(
                k in data.columns and self._match_criterion(data[k].iloc[0], v)
                for k, v in criteria.items()
            )
        return False

    def create_checkpoint(self, name: str, include_cache: bool = False) -> Dict[str, Any]:
        """Create a checkpoint of current state"""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "data_store": self.data_store,
                "metadata_store": self.metadata_store,
                "index_store": self.index_store if self.config["index_enabled"] else {},
                "operation_history": list(self.operation_history),
                "cache": self.cache if include_cache else {},
                "config": self.config
            }

            checkpoint_path = Path(f"data/checkpoints/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.checkpoints[name] = str(checkpoint_path)

            # Create backup metadata
            checkpoint_metadata = {
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "size": os.path.getsize(checkpoint_path),
                "includes_cache": include_cache,
                "path": str(checkpoint_path)
            }

            self._add_to_operation_history("checkpoint_create", name, checkpoint_metadata)

            return {
                "success": True,
                "message": "Checkpoint created successfully",
                "metadata": checkpoint_metadata
            }

        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {str(e)}")
            return {"success": False, "error": str(e)}

    def restore_checkpoint(self, name: str, include_cache: bool = True) -> Dict[str, Any]:
        """Restore state from a checkpoint"""
        try:
            if name not in self.checkpoints:
                return {"success": False, "error": "Checkpoint not found"}

            checkpoint_path = self.checkpoints[name]

            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            with self.locks["data"]:
                self.data_store = checkpoint_data["data_store"]

            with self.locks["metadata"]:
                self.metadata_store = checkpoint_data["metadata_store"]

            if self.config["index_enabled"]:
                with self.locks["index"]:
                    self.index_store = checkpoint_data["index_store"]

            if include_cache and "cache" in checkpoint_data:
                with self.locks["cache"]:
                    self.cache = checkpoint_data["cache"]

            self.operation_history = deque(
                checkpoint_data["operation_history"],
                maxlen=self.config["max_history_size"]
            )

            # Merge configurations, keeping current settings for critical parameters
            critical_config = {
                "async_operations": self.config["async_operations"],
                "max_workers": self.config["max_workers"]
            }
            self.config.update(checkpoint_data["config"])
            self.config.update(critical_config)

            self._add_to_operation_history(
                "checkpoint_restore",
                name,
                {"timestamp": datetime.now().isoformat()}
            )

            return {"success": True, "message": "Checkpoint restored successfully"}

        except Exception as e:
            self.logger.error(f"Error restoring checkpoint: {str(e)}")
            return {"success": False, "error": str(e)}

    def list_checkpoints(self) -> Dict[str, Any]:
        """List all available checkpoints"""
        try:
            checkpoint_info = {}
            for name, path in self.checkpoints.items():
                if os.path.exists(path):
                    checkpoint_info[name] = {
                        "path": path,
                        "size": os.path.getsize(path),
                        "created": datetime.fromtimestamp(
                            os.path.getctime(path)
                        ).isoformat(),
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(path)
                        ).isoformat()
                    }

            return {
                "success": True,
                "checkpoints": checkpoint_info
            }

        except Exception as e:
            self.logger.error(f"Error listing checkpoints: {str(e)}")
            return {"success": False, "error": str(e)}

    def _compress_data(self, data: Any) -> bytes:
        """Compress data using the configured compression level"""
        import zlib
        try:
            serialized_data = pickle.dumps(data)
            return zlib.compress(serialized_data, self.config["compression_level"])
        except Exception as e:
            self.logger.error(f"Error compressing data: {str(e)}")
            return data

    def _decompress_data(self, data: Any) -> Any:
        """Decompress data"""
        import zlib
        try:
            if isinstance(data, bytes):
                decompressed_data = zlib.decompress(data)
                return pickle.loads(decompressed_data)
            return data
        except Exception as e:
            self.logger.error(f"Error decompressing data: {str(e)}")
            return data

    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        try:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, bytes):
                return len(data)
            else:
                return len(pickle.dumps(data))
        except:
            return 0

    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash of data"""
        try:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
            elif isinstance(data, bytes):
                return hashlib.md5(data).hexdigest()
            else:
                return hashlib.md5(pickle.dumps(data)).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash: {str(e)}")
            return ""

    def _update_index(self, key: str, data: Any):
        """Update search index for data"""
        try:
            with self.locks["index"]:
                if isinstance(data, dict):
                    self.index_store[key] = {
                        "content_hash": self._calculate_hash(data),
                        "fields": list(data.keys()),
                        "size": self._calculate_size(data),
                        "type": "dict"
                    }
                elif isinstance(data, pd.DataFrame):
                    self.index_store[key] = {
                        "content_hash": self._calculate_hash(data),
                        "columns": list(data.columns),
                        "size": self._calculate_size(data),
                        "type": "dataframe",
                        "row_count": len(data)
                    }
                elif isinstance(data, pd.Series):
                    self.index_store[key] = {
                        "content_hash": self._calculate_hash(data),
                        "name": data.name,
                        "size": self._calculate_size(data),
                        "type": "series",
                        "length": len(data)
                    }
        except Exception as e:
            self.logger.error(f"Error updating index: {str(e)}")

    def _rebuild_index(self):
        """Rebuild the entire search index"""
        try:
            with self.locks["index"]:
                self.index_store.clear()
                for key, data in self.data_store.items():
                    self._update_index(key, data)
            self.logger.info("Index rebuilt successfully")
        except Exception as e:
            self.logger.error(f"Error rebuilding index: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            stats = {
                "total_entries": len(self.data_store),
                "total_size": sum(self._calculate_size(data) for data in self.data_store.values()),
                "operation_count": len(self.operation_history),
                "metadata_count": len(self.metadata_store),
                "index_count": len(self.index_store),
                "checkpoint_count": len(self.checkpoints),
                "cache_entries": len(self.cache),
                "cache_size": sum(self._calculate_size(entry["data"]) for entry in self.cache.values()),
                "system_status": {
                    "running": self._running,
                    "async_operations": self.config["async_operations"],
                    "index_enabled": self.config["index_enabled"],
                    "compression_enabled": self.config["compression_enabled"]
                }
            }
            return {"success": True, "stats": stats}
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}")
            return {"success": False, "error": str(e)}

    def cleanup(self) -> Dict[str, Any]:
        """Perform system cleanup"""
        try:
            # Clean up old checkpoints
            checkpoint_dir = Path("data/checkpoints")
            removed_checkpoints = 0
            for checkpoint_file in checkpoint_dir.glob("*.pkl"):
                if (datetime.now() - datetime.fromtimestamp(checkpoint_file.stat().st_mtime)).days > self.config[
                    "max_checkpoint_age"]:
                    checkpoint_file.unlink()
                    removed_checkpoints += 1

            # Clean up temporary files
            temp_dir = Path("data/temp")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                temp_dir.mkdir()

            # Clean up old cache entries
            with self.locks["cache"]:
                current_time = time.time()
                expired_cache = [
                    k for k, v in self.cache.items()
                    if current_time - v["timestamp"] > self.config["cache_ttl"]
                ]
                for key in expired_cache:
                    del self.cache[key]

            cleanup_stats = {
                "removed_checkpoints": removed_checkpoints,
                "cleared_temp": True,
                "removed_cache_entries": len(expired_cache)
            }

            self._add_to_operation_history("cleanup", None, cleanup_stats)

            return {
                "success": True,
                "message": "Cleanup completed successfully",
                "stats": cleanup_stats
            }

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return {"success": False, "error": str(e)}

    def shutdown(self):
        """Shutdown the memory system gracefully"""
        try:
            self._running = False

            # Wait for background tasks to complete
            threads = [
                getattr(self, name, None)
                for name in ['operation_thread', 'backup_thread', 'cleanup_thread', 'cache_thread', 'index_thread']
            ]

            for thread in threads:
                if thread is not None:
                    thread.join(timeout=5)

            # Final backup if enabled
            if self.config["backup_enabled"]:
                self.create_backup()

            # Final cleanup
            self.cleanup()

            # Shutdown thread pool
            self.executor.shutdown(wait=True)

            self.logger.info("Memory system shut down successfully")
            return {"success": True, "message": "Memory system shut down successfully"}
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update memory system configuration"""
        try:
            # Validate critical configurations before updating
            critical_configs = {
                "max_workers", "async_operations", "backup_enabled",
                "index_enabled", "compression_enabled"
            }

            # Only update non-critical configurations or validated critical ones
            for key, value in new_config.items():
                if key not in critical_configs or self._validate_config_update(key, value):
                    self.config[key] = value

            self._add_to_operation_history(
                "config_update",
                None,
                {"updated_keys": list(new_config.keys())}
            )

            return {
                "success": True,
                "message": "Configuration updated successfully",
                "updated_config": self.config
            }

        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def _validate_config_update(self, key: str, value: Any) -> bool:
        """Validate configuration updates for critical settings"""
        try:
            if key == "max_workers":
                return isinstance(value, int) and value > 0
            elif key in {"async_operations", "backup_enabled", "index_enabled", "compression_enabled"}:
                return isinstance(value, bool)
            return True
        except Exception:
            return False

    def create_backup(self) -> Dict[str, Any]:
        """Create a backup of the current state"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = Path(f"data/backups/backup_{timestamp}")
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup core data
            with open(backup_path / "data_store.pkl", 'wb') as f:
                pickle.dump(self.data_store, f)

            with open(backup_path / "metadata_store.pkl", 'wb') as f:
                pickle.dump(self.metadata_store, f)

            # Backup configuration and operational data
            backup_info = {
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "checkpoints": self.checkpoints,
                "stats": self.get_stats()["stats"] if self.get_stats()["success"] else {}
            }

            with open(backup_path / "backup_info.json", 'w') as f:
                json.dump(backup_info, f, indent=2, default=str)

            self._add_to_operation_history(
                "backup_create",
                None,
                {"backup_path": str(backup_path)}
            )

            return {
                "success": True,
                "message": "Backup created successfully",
                "backup_path": str(backup_path),
                "backup_info": backup_info
            }

        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return {"success": False, "error": str(e)}

    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files"""
        try:
            backup_dir = Path("data/backups")
            if not backup_dir.exists():
                return

            backups = sorted(
                backup_dir.glob("backup_*"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Keep only the specified number of recent backups
            backups_to_remove = backups[self.config["backup_retention"]:]
            for backup in backups_to_remove:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()

        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {str(e)}")

    def _add_to_operation_history(self, operation_type: str, key: Optional[str],
                                  details: Optional[Dict[str, Any]] = None) -> None:
        """Add an operation to the history"""
        try:
            operation = {
                "timestamp": datetime.now().isoformat(),
                "type": operation_type,
                "key": key,
                "details": details or {}
            }

            with self.locks["operation"]:
                self.operation_history.append(operation)

        except Exception as e:
            self.logger.error(f"Error adding to operation history: {str(e)}")

    def summarize_current_context(self) -> Dict[str, Any]:
        """Summarize the current context of the memory system"""
        try:
            summary = {
                "total_entries": len(self.data_store),
                "total_operations": len(self.operation_history),
                "recent_operations": list(self.operation_history)[-5:],
                "active_checkpoints": len(self.checkpoints),
                "cache_status": {
                    "entries": len(self.cache),
                    "size": sum(self._calculate_size(entry["data"]) for entry in self.cache.values())
                },
                "index_status": {
                    "entries": len(self.index_store),
                    "last_updated": max(
                        (entry.get("timestamp", "1970-01-01T00:00:00")
                         for entry in self.index_store.values()),
                        default="No entries"
                    )
                }
            }

            return {"success": True, "summary": summary}

        except Exception as e:
            self.logger.error(f"Error summarizing context: {str(e)}")
            return {"success": False, "error": str(e)}

    def add_operation(self, operation_type: str, details: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Record an operation in the memory system"""
        try:
            operation = {
                "timestamp": datetime.now().isoformat(),
                "type": operation_type,
                "details": details,
                "result": result
            }

            with self.locks["operation"]:
                self.operation_history.append(operation)

            # Update metadata if the operation involves data manipulation
            if operation_type in ["data_load", "data_transform", "visualization"]:
                self._update_operation_metadata(operation)

            return {"success": True, "operation_id": len(self.operation_history)}

        except Exception as e:
            self.logger.error(f"Error adding operation: {str(e)}")
            return {"success": False, "error": str(e)}

    def log_exception(self, exception: Exception) -> None:
        """Log an exception with detailed information"""
        try:
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }

            # Log to file
            self.logger.error(f"Exception occurred: {error_info['type']}: {error_info['message']}")
            self.logger.error(f"Traceback: {error_info['traceback']}")

            # Add to operation history
            self.add_operation(
                operation_type="exception",
                details=error_info,
                result=None
            )

        except Exception as e:
            self.logger.error(f"Error logging exception: {str(e)}")

    def _update_operation_metadata(self, operation: Dict[str, Any]) -> None:
        """Update metadata based on operation type"""
        try:
            operation_type = operation["type"]
            timestamp = operation["timestamp"]

            metadata = {
                "last_operation": operation_type,
                "last_operation_time": timestamp,
                "operation_count": self.metadata_store.get("operation_count", 0) + 1
            }

            with self.locks["metadata"]:
                if "system_metadata" not in self.metadata_store:
                    self.metadata_store["system_metadata"] = {}
                self.metadata_store["system_metadata"].update(metadata)

        except Exception as e:
            self.logger.error(f"Error updating operation metadata: {str(e)}")