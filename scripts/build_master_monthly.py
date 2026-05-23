#!/usr/bin/env python3
"""
Main script to run the monthly financial data preprocessing pipeline.

This script instantiates the MonthlyDataProcessor class with the configuration
for the input files and runs the full pipeline to produce cleaned individual
series files and a merged master dataset.

Usage:
    python scripts/build_master_monthly.py
"""

import sys
import pathlib
from pathlib import Path

# Add current directory to path for importing src package
sys.path.insert(0, '.')

# Import from src package
from src.monthly_data_processor import MonthlyDataProcessor


def main():
    # Configuration: map input file names to series info
    config = {
        'dollar_1mo_data.csv': {'name': 'dollar', 'end_date': '2025-12-01'},
        'vix_1mo_data.csv': {'name': 'vix', 'end_date': '2025-12-01'},
        'STLFSI4_1mo_data.csv': {'name': 'STLFSI4', 'end_date': '2025-12-01'},
        'vxy_g7_1mo_data.xlsx': {'name': 'vxy_g7', 'end_date': '2025-12-01'},
        'vxy_em_1mo_data.xlsx': {'name': 'vxy_em', 'end_date': '2025-12-01'},
        'gepu_1mo_data.xlsx': {'name': 'gepu', 'end_date': '2025-11-01'},
    }

    # Define directories relative to the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_dir = project_root / 'data' / 'raw'
    interim_dir = project_root / 'data' / 'interim'
    processed_dir = project_root / 'data' / 'processed'

    # Create output directories if they don't exist
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run the processor
    processor = MonthlyDataProcessor(config)
    processor.run(input_dir, interim_dir, processed_dir)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()