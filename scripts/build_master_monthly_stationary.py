#!/usr/bin/env python3
"""
Thin runner script for building the stationary master monthly dataset.

This script imports from src/stationarity_processor and runs the full pipeline
to create transformations, run ADF tests, and save results.
"""

import sys
from pathlib import Path

# Add current directory to path for importing src package
sys.path.insert(0, '.')

from src.stationarity_processor import StationarityProcessor

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    master_file = project_root / "data" / "processed" / "master_monthly.csv"
    output_dir = project_root / "data" / "processed"
    tables_dir = project_root / "output" / "tables"

    # Initialize and run
    processor = StationarityProcessor(master_file, output_dir, tables_dir)
    processor.run_full_pipeline()

if __name__ == "__main__":
    main()

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    master_file = project_root / "data" / "processed" / "master_monthly.csv"
    output_dir = project_root / "data" / "processed"
    tables_dir = project_root / "output" / "tables"

    # Initialize and run
    processor = StationarityProcessor(master_file, output_dir, tables_dir)
    processor.run_full_pipeline()

if __name__ == "__main__":
    main()