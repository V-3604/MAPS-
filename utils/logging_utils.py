import logging
import logging.config
import os
import sys
import json
import traceback
from datetime import datetime


def setup_logging(log_config, default_level=logging.INFO):
    """Set up logging configuration"""
    if isinstance(log_config, dict):
        logging.config.dictConfig(log_config)
    elif os.path.exists(log_config):
        with open(log_config, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    return logging.getLogger(__name__)


def log_operation(logger, operation_type, details, success=True, result=None, error=None):
    """Log an operation with standardized format"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "operation_type": operation_type,
        "details": details,
        "success": success
    }

    if result is not None:
        log_data["result"] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)

    if error is not None:
        log_data["error"] = str(error)

    if success:
        logger.info(f"Operation: {operation_type} - {details}")
    else:
        logger.error(f"Operation failed: {operation_type} - {details} - Error: {error}")

    return log_data


def log_exception(logger, exc_info=None):
    """Log exception details"""
    if exc_info is None:
        exc_info = sys.exc_info()

    exc_type, exc_value, exc_traceback = exc_info

    logger.error("Exception occurred:", exc_info=True)

    # Format traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = ''.join(tb_lines)

    # Log detailed error
    logger.debug(f"Exception details:\n{tb_text}")

    return {
        "timestamp": datetime.now().isoformat(),
        "exception_type": str(exc_type.__name__),
        "exception_message": str(exc_value),
        "traceback": tb_text
    }


def create_session_logger(session_id, log_dir):
    """Create a logger for a specific session"""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name
    log_file = os.path.join(log_dir, f"session_{session_id}.log")

    # Create logger
    logger = logging.getLogger(f"session_{session_id}")
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )

    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_system_info(logger):
    """Log system information"""
    import platform

    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "memory": "Unknown"  # Would require additional libraries like psutil
    }

    # Try to get memory info if psutil is available
    try:
        import psutil
        mem = psutil.virtual_memory()
        system_info["memory"] = f"{mem.total / (1024 ** 3):.1f} GB"
    except ImportError:
        pass

    logger.info(f"System information: {json.dumps(system_info)}")
    return system_info