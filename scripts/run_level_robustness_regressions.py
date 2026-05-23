#!/usr/bin/env python3
"""
Thin runner script for appendix-level level-based robustness regressions.

This script imports from src.regression_processor and runs the robustness
workflow that uses levels for regressors found to be stationary in levels while
keeping dollar_log_return and gepu_diff in transformed form.
"""

import sys
from pathlib import Path

# Add current directory to path for importing src package
sys.path.insert(0, '.')

from src.regression_processor import RegressionProcessor


def main():
    project_root = Path(__file__).parent.parent
    stationary_file = project_root / "data" / "processed" / "master_monthly_stationary.csv"
    level_file = project_root / "data" / "processed" / "master_monthly.csv"
    tables_dir = project_root / "output" / "tables"

    processor = RegressionProcessor(
        stationary_file=stationary_file,
        tables_dir=tables_dir,
        level_file=level_file,
    )
    processor.run_level_robustness_analysis()


if __name__ == "__main__":
    main()
