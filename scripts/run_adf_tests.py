#!/usr/bin/env python3
"""
Thin runner script for running ADF tests only.

This script imports from src/stationarity_processor and runs only the ADF test
and export part, without creating the stationary dataset.
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

    # Initialize and run ADF only
    processor = StationarityProcessor(master_file, output_dir, tables_dir)
    processor.run_adf_only()

if __name__ == "__main__":
    main()

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    master_file = project_root / "data" / "processed" / "master_monthly.csv"
    output_dir = project_root / "data" / "processed"
    tables_dir = project_root / "output" / "tables"

    # Initialize and run ADF only
    processor = StationarityProcessor(master_file, output_dir, tables_dir)
    processor.run_adf_only()

if __name__ == "__main__":
    main()