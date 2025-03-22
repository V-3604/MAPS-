# init_setup.py
import os
import sys
from pathlib import Path


def create_project_structure():
    """Create the initial project structure with all required directories"""
    base_dir = Path(__file__).parent.parent

    directories = [
        "agents",
        "config",
        "core",
        "data/raw",
        "data/processed",
        "data/temp",
        "data/sample_datasets",
        "data/memory",
        "data/conversations",
        "data/visualizations",
        "data/checkpoints",
        "examples",
        "notebooks",
        "output/visualizations",
        "output/reports",
        "output/logs",
        "tests",
        "utils"
    ]

    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

    print("Project structure created successfully!")


if __name__ == "__main__":
    create_project_structure()