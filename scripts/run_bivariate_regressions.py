#!/usr/bin/env python3
"""
Thin runner script for running bivariate regressions on stationary data.

This script imports from src/regression_processor and runs the bivariate
regression analysis pipeline.
"""

import sys
from pathlib import Path

# Add current directory to path for importing src package
sys.path.insert(0, '.')

from src.regression_processor import RegressionProcessor

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    stationary_file = project_root / "data" / "processed" / "master_monthly_stationary.csv"
    tables_dir = project_root / "output" / "tables"

    # Initialize and run
    processor = RegressionProcessor(stationary_file, tables_dir)
    processor.run_full_analysis()

if __name__ == "__main__":
    main()