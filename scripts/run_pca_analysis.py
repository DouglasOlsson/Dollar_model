#!/usr/bin/env python3
"""
Thin runner for PCA analysis.
"""
import sys
from pathlib import Path

# Ensure src/ can be imported when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pca_processor import PCAProcessor


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stationary_file = root / 'data' / 'processed' / 'master_monthly_stationary.csv'
    tables_dir = root / 'output' / 'tables'
    processed_dir = root / 'data' / 'processed'

    processor = PCAProcessor(
        stationary_file=stationary_file,
        tables_dir=tables_dir,
        processed_dir=processed_dir,
    )

    processor.run_full_pipeline()
    print('PCA analysis complete.')
    print('Saved PCA explained variance, loadings, scores, and correlations for specifications A, B, and C.')
    print('Saved PCA regression diagnostics in output/tables/pca_regressions.csv and LaTeX output.')


if __name__ == '__main__':
    main()
