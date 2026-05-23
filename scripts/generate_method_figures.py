#!/usr/bin/env python3
"""
Thin runner for generating methodology figures.
"""
import sys
from pathlib import Path

# Add current directory to path for importing src package
sys.path.insert(0, '.')

from src.plotting_processor import PlottingProcessor


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    raw_file = project_root / 'data' / 'processed' / 'master_monthly.csv'
    stationary_file = project_root / 'data' / 'processed' / 'master_monthly_stationary.csv'
    figures_dir = project_root / 'figures'

    processor = PlottingProcessor(
        raw_file=raw_file,
        stationary_file=stationary_file,
        figures_dir=figures_dir,
    )

    processor.generate_method_figures()
    print('Methodology figures generated successfully.')
    print(f'Figures saved to {figures_dir}')


if __name__ == '__main__':
    main()
