import os

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Configure data directories
DATA_DIRS = {
    "raw": os.path.join(DATA_DIR, "raw"),
    "processed": os.path.join(DATA_DIR, "processed"),
    "temp": os.path.join(DATA_DIR, "temp"),
    "sample_datasets": os.path.join(DATA_DIR, "sample_datasets"),
    "memory": os.path.join(DATA_DIR, "memory"),
    "conversations": os.path.join(DATA_DIR, "conversations"),
    "visualizations": os.path.join(DATA_DIR, "visualizations"),
    "checkpoints": os.path.join(DATA_DIR, "checkpoints")
}

# Configure output directories
OUTPUT_DIRS = {
    "visualizations": os.path.join(OUTPUT_DIR, "visualizations"),
    "reports": os.path.join(OUTPUT_DIR, "reports"),
    "logs": os.path.join(OUTPUT_DIR, "logs")
}

# Create all required directories
for dir_path in list(DATA_DIRS.values()) + list(OUTPUT_DIRS.values()):
    os.makedirs(dir_path, exist_ok=True)

# System settings
SYSTEM_SETTINGS = {
    "debug_mode": True,  # Enable/disable debug information
    "log_level": "INFO",  # Logging level (DEBUG, INFO, WARNING, ERROR)
    "max_dataframe_size": 1000000,  # Maximum number of cells in dataframe (rows*cols)
    "max_file_size_mb": 100,  # Maximum file size for data loading (in MB)
    "timeout": 30,  # Operation timeout in seconds
    "save_intermediate_results": True  # Whether to save intermediate results during processing
}

# Logging configuration
LOG_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": os.path.join(OUTPUT_DIRS["logs"], "maps.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True
        }
    }
}