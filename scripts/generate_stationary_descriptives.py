#!/usr/bin/env python3
"""
Thin runner script for generating descriptive analysis on stationary data.

This script imports from src/stationarity_processor and runs the descriptive
analysis pipeline to produce stats, correlations, and LaTeX outputs.
"""

import sys
from pathlib import Path

# Add current directory to path for importing src package
sys.path.insert(0, '.')

from src.stationarity_processor import StationarityProcessor

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    master_file = project_root / "data" / "processed" / "master_monthly.csv"  # Not used, but for class init
    output_dir = project_root / "data" / "processed"
    tables_dir = project_root / "output" / "tables"

    # Initialize and run descriptive analysis
    processor = StationarityProcessor(master_file, output_dir, tables_dir)
    processor.run_descriptive_analysis()

if __name__ == "__main__":
    main()