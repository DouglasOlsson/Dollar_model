#!/usr/bin/env python3
"""
Thin runner script for supplementary structural break appendix tests.

This script imports from src.structural_break_processor and runs both the
fixed-boundary Chow tests and the separate exploratory breakpoint search.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.structural_break_processor import StructuralBreakProcessor


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    stationary_file = project_root / 'data' / 'processed' / 'master_monthly_stationary.csv'
    tables_dir = project_root / 'output' / 'tables'

    processor = StructuralBreakProcessor(
        stationary_file=stationary_file,
        tables_dir=tables_dir,
    )
    processor.run_full_analysis()

    print('Structural break appendix tests complete.')
    print('Saved fixed-break and data-driven search outputs to output/tables/.')


if __name__ == '__main__':
    main()
