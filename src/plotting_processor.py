#!/usr/bin/env python3
"""
Plotting Processor Module

This module contains PlottingProcessor, which creates methodology figures showing
raw series levels and their transformed, stationary counterparts.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


class PlottingProcessor:
    """
    A class to generate methodology figures comparing raw series levels to
    stationary transformations.
    """

    def __init__(self, raw_file: Path, stationary_file: Path, figures_dir: Path):
        """
        Initialize the plotting processor.

        Args:
            raw_file: Path to the raw master monthly CSV file.
            stationary_file: Path to the stationary master monthly CSV file.
            figures_dir: Directory where figure PNG files will be saved.
        """
        self.raw_file = raw_file
        self.stationary_file = stationary_file
        self.figures_dir = figures_dir
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.stationary_data: pd.DataFrame = pd.DataFrame()

    def load_raw_data(self) -> None:
        """
        Load the raw master monthly dataset with parsed Date and sorted index.
        """
        if not self.raw_file.exists():
            raise FileNotFoundError(f"Raw file not found: {self.raw_file}")
        self.raw_data = pd.read_csv(self.raw_file)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        self.raw_data = self.raw_data.sort_values('Date').reset_index(drop=True)

    def load_stationary_data(self) -> None:
        """
        Load the stationary dataset with parsed Date and sorted index.
        """
        if not self.stationary_file.exists():
            raise FileNotFoundError(f"Stationary file not found: {self.stationary_file}")
        self.stationary_data = pd.read_csv(self.stationary_file)
        self.stationary_data['Date'] = pd.to_datetime(self.stationary_data['Date'])
        self.stationary_data = self.stationary_data.sort_values('Date').reset_index(drop=True)

    def _prepare_plot_series(self, raw_variable: str, transformed_variable: str) -> Dict[str, pd.Series]:
        """
        Extract and align series for plotting.

        Args:
            raw_variable: Column name in the raw dataset.
            transformed_variable: Column name in the stationary dataset.

        Returns:
            A dictionary with aligned raw and transformed series.
        """
        if self.raw_data.empty or self.stationary_data.empty:
            raise ValueError('Data must be loaded before creating plots.')

        raw_series = self.raw_data[['Date', raw_variable]].dropna()
        transformed_series = self.stationary_data[['Date', transformed_variable]].dropna()

        raw_series = raw_series.set_index('Date')[raw_variable]
        transformed_series = transformed_series.set_index('Date')[transformed_variable]

        return {'raw': raw_series, 'transformed': transformed_series}

    def plot_variable(self, raw_variable: str, transformed_variable: str, title: str, output_file: Path) -> None:
        """
        Create a two-panel figure for a raw variable and its transformed counterpart.

        Args:
            raw_variable: Column name in the raw dataset.
            transformed_variable: Column name in the stationary dataset.
            title: Figure title.
            output_file: Output path for the PNG figure.
        """
        series = self._prepare_plot_series(raw_variable, transformed_variable)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        axs[0].plot(series['raw'].index, series['raw'].values)
        axs[0].set_title('Level series')
        axs[0].set_ylabel(raw_variable)

        axs[1].plot(series['transformed'].index, series['transformed'].values)
        axs[1].set_title('Transformed series')
        axs[1].set_ylabel(transformed_variable)
        axs[1].set_xlabel('Date')

        for ax in axs:
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_file, dpi=300)
        plt.close(fig)

    def generate_method_figures(self) -> None:
        """
        Generate all methodology figures for raw and transformed series.
        """
        self.load_raw_data()
        self.load_stationary_data()

        plot_specs: List[Dict[str, str]] = [
            {
                'raw': 'dollar',
                'transformed': 'dollar_log_return',
                'filename': 'dollar_level_vs_log_return.png',
                'title': 'Dollar Index: Level and Log Return',
            },
            {
                'raw': 'vix',
                'transformed': 'vix_diff',
                'filename': 'vix_level_vs_diff.png',
                'title': 'VIX: Level and First Difference',
            },
            {
                'raw': 'STLFSI4',
                'transformed': 'STLFSI4_diff',
                'filename': 'STLFSI4_level_vs_diff.png',
                'title': 'STLFSI4: Level and First Difference',
            },
            {
                'raw': 'vxy_g7',
                'transformed': 'vxy_g7_diff',
                'filename': 'vxy_g7_level_vs_diff.png',
                'title': 'VXY_G7: Level and First Difference',
            },
            {
                'raw': 'vxy_em',
                'transformed': 'vxy_em_diff',
                'filename': 'vxy_em_level_vs_diff.png',
                'title': 'VXY_EM: Level and First Difference',
            },
            {
                'raw': 'gepu',
                'transformed': 'gepu_diff',
                'filename': 'gepu_level_vs_diff.png',
                'title': 'GEPU: Level and First Difference',
            },
        ]

        for spec in plot_specs:
            output_file = self.figures_dir / spec['filename']
            self.plot_variable(
                raw_variable=spec['raw'],
                transformed_variable=spec['transformed'],
                title=spec['title'],
                output_file=output_file,
            )
