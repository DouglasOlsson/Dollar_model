#!/usr/bin/env python3
"""
Stationarity Processor Module

This module contains the StationarityProcessor class for handling stationarity
transformations and ADF tests on the master monthly dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from typing import List, Dict, Optional


class StationarityProcessor:
    """
    A class for performing stationarity transformations and ADF tests on monthly data.

    This class loads the master monthly dataset, applies transformations to achieve
    stationarity (log returns for dollar, first differences for others), runs ADF tests,
    and exports results and stationary datasets.
    """

    def __init__(self, master_file: Path, output_dir: Path, tables_dir: Path):
        """
        Initialize the processor.

        Args:
            master_file: Path to the master monthly CSV file.
            output_dir: Directory for processed data outputs.
            tables_dir: Directory for table outputs (e.g., ADF results).
        """
        self.master_file = master_file
        self.output_dir = output_dir
        self.tables_dir = tables_dir
        self.data: Optional[pd.DataFrame] = None
        self.transformed_data: Optional[pd.DataFrame] = None
        self.adf_results: List[Dict] = []

    def load_data(self) -> None:
        """
        Load the master monthly data, parse dates, and sort.
        """
        if not self.master_file.exists():
            raise FileNotFoundError(f"Master file {self.master_file} does not exist.")
        self.data = pd.read_csv(self.master_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)

    def create_transforms(self) -> None:
        """
        Create stationary transformations: log return for dollar, differences for others.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.transformed_data = self.data.copy()
        self.transformed_data['dollar_log_return'] = np.log(self.data['dollar']).diff()
        self.transformed_data['vix_diff'] = self.data['vix'].diff()
        self.transformed_data['STLFSI4_diff'] = self.data['STLFSI4'].diff()
        self.transformed_data['vxy_g7_diff'] = self.data['vxy_g7'].diff()
        self.transformed_data['vxy_em_diff'] = self.data['vxy_em'].diff()
        self.transformed_data['gepu_diff'] = self.data['gepu'].diff()

    def run_adf_tests(self) -> None:
        """
        Run ADF tests on level and transformed variables.
        """
        if self.data is None or self.transformed_data is None:
            raise ValueError("Data not loaded or transformed. Call load_data() and create_transforms() first.")

        variables_level = ['dollar', 'vix', 'STLFSI4', 'vxy_g7', 'vxy_em', 'gepu']
        variables_transformed = ['dollar_log_return', 'vix_diff', 'STLFSI4_diff', 'vxy_g7_diff', 'vxy_em_diff', 'gepu_diff']

        # Test levels
        for var in variables_level:
            result = self._run_single_adf(self.data[var], var, 'level')
            if result:
                self.adf_results.append(result)

        # Test transformed
        for var in variables_transformed:
            result = self._run_single_adf(self.transformed_data[var], var, 'transformed')
            if result:
                self.adf_results.append(result)

    def _run_single_adf(self, series: pd.Series, name: str, form: str) -> Optional[Dict]:
        """
        Run a single ADF test.

        Args:
            series: Series to test.
            name: Name of the series.
            form: Form tested ('level' or 'transformed').

        Returns:
            Dict with test results or None if error.
        """
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            return {
                'series': name,
                'form_tested': form,
                'adf_statistic': result[0],
                'p_value': result[1],
                'used_lag': result[2],
                'n_obs': result[3],
                'stationary_5pct': result[4]['5%'],
                'decision': "Stationary" if result[1] < 0.05 else "Non-stationary"
            }
        except Exception as e:
            print(f"Error running ADF on {name} ({form}): {e}")
            return None

    def save_adf_results(self) -> None:
        """
        Save ADF results to CSV.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        adf_df = pd.DataFrame(self.adf_results)
        adf_file = self.tables_dir / "adf_results_stationarity.csv"
        adf_df.to_csv(adf_file, index=False)
        print(f"ADF results saved to {adf_file}")

    def save_stationary_dataset(self) -> None:
        """
        Save the stationary dataset, dropping the first row due to transformations.
        """
        if self.transformed_data is None:
            raise ValueError("Transformed data not available. Call create_transforms() first.")
        stationary_columns = ['Date'] + ['dollar_log_return', 'vix_diff', 'STLFSI4_diff', 'vxy_g7_diff', 'vxy_em_diff', 'gepu_diff']
        df_stationary = self.transformed_data[stationary_columns].iloc[1:].reset_index(drop=True)
        stationary_file = self.output_dir / "master_monthly_stationary.csv"
        df_stationary.to_csv(stationary_file, index=False)
        print(f"Stationary dataset saved to {stationary_file} with {len(df_stationary)} rows.")

    def run_full_pipeline(self) -> None:
        """
        Run the full stationarity pipeline: load, transform, test, save.
        """
        self.load_data()
        self.create_transforms()
        self.run_adf_tests()
        self.save_adf_results()
        self.save_stationary_dataset()
        print(f"Processed {len(self.data)} original rows into {len(self.transformed_data.iloc[1:])} stationary rows.")

    def load_stationary_data(self) -> None:
        """
        Load the stationary dataset, parse dates, and sort.
        """
        stationary_file = self.output_dir / "master_monthly_stationary.csv"
        if not stationary_file.exists():
            raise FileNotFoundError(f"Stationary file {stationary_file} does not exist.")
        self.data = pd.read_csv(stationary_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)

    def generate_descriptive_stats(self) -> pd.DataFrame:
        """
        Generate descriptive statistics for the stationary variables.

        Returns:
            DataFrame with mean, std, min, max, count.
        """
        if self.data is None:
            raise ValueError("Stationary data not loaded. Call load_stationary_data() first.")
        variables = ['dollar_log_return', 'vix_diff', 'STLFSI4_diff', 'vxy_g7_diff', 'vxy_em_diff', 'gepu_diff']
        stats = self.data[variables].describe().T[['mean', 'std', 'min', 'max', 'count']]
        return stats

    def generate_correlation_matrix(self) -> pd.DataFrame:
        """
        Generate the full Pearson correlation matrix for stationary variables.

        Returns:
            DataFrame with correlation matrix.
        """
        if self.data is None:
            raise ValueError("Stationary data not loaded. Call load_stationary_data() first.")
        variables = ['dollar_log_return', 'vix_diff', 'STLFSI4_diff', 'vxy_g7_diff', 'vxy_em_diff', 'gepu_diff']
        corr_matrix = self.data[variables].corr()
        return corr_matrix

    def generate_correlation_rank_to_dollar(self) -> pd.DataFrame:
        """
        Generate a ranking of absolute correlations with dollar_log_return, excluding itself.

        Returns:
            DataFrame with variable and abs_corr, sorted descending.
        """
        if self.data is None:
            raise ValueError("Stationary data not loaded. Call load_stationary_data() first.")
        variables = ['vix_diff', 'STLFSI4_diff', 'vxy_g7_diff', 'vxy_em_diff', 'gepu_diff']
        corr_with_dollar = self.data[variables + ['dollar_log_return']].corr()['dollar_log_return'][variables]
        rank_df = pd.DataFrame({
            'variable': corr_with_dollar.index,
            'abs_corr': corr_with_dollar.abs()
        }).sort_values('abs_corr', ascending=False)
        return rank_df

    def export_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export DataFrame to CSV in tables_dir.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename
        df.to_csv(file_path, index=True)
        print(f"Exported {filename} to {file_path}")

    def export_latex(self, df: pd.DataFrame, filename: str, caption: str = "", label: str = "", index: bool = True) -> None:
        """
        Export DataFrame to LaTeX table format, rounded to 3 decimals.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename
        # Round to 3 decimals
        df_rounded = df.round(3)
        # Generate LaTeX table manually
        latex_str = self._df_to_latex(df_rounded, caption, label, index)
        with open(file_path, 'w') as f:
            f.write(latex_str)
        print(f"Exported {filename} to {file_path}")

    def _df_to_latex(self, df: pd.DataFrame, caption: str, label: str, index: bool) -> str:
        """
        Convert DataFrame to LaTeX table string manually.
        """
        lines = []
        num_cols = len(df.columns) + (1 if index else 0)
        if caption:
            lines.append(f"\\begin{{table}}[H]")
            lines.append(f"\\caption{{{caption}}}")
            if label:
                lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{" + "c" * num_cols + "}")

        # Header
        if index:
            header = f"{df.index.name or 'Index'} & " + " & ".join(df.columns)
        else:
            header = " & ".join(df.columns)
        lines.append("\\hline")
        lines.append(header + " \\\\")
        lines.append("\\hline")

        # Rows
        for idx, row in df.iterrows():
            if index:
                row_values = [idx] + list(row.values)
            else:
                row_values = list(row.values)
            formatted_values = []
            for val in row_values:
                if isinstance(val, (int, float)):
                    formatted_values.append(f"{val:.3f}")
                else:
                    formatted_values.append(str(val))
            row_str = " & ".join(formatted_values)
            lines.append(row_str + " \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        if caption:
            lines.append("\\end{table}")

        return "\n".join(lines)

    def run_descriptive_analysis(self) -> None:
        """
        Run the full descriptive analysis: load data, compute stats, correlations, ranks, and export.
        """
        self.load_stationary_data()
        desc_stats = self.generate_descriptive_stats()
        corr_matrix = self.generate_correlation_matrix()
        corr_rank = self.generate_correlation_rank_to_dollar()

        # Export CSVs
        self.export_csv(desc_stats, "stationary_descriptive_stats.csv")
        self.export_csv(corr_matrix, "stationary_correlation_matrix.csv")
        self.export_csv(corr_rank, "stationary_correlation_rank_to_dollar.csv")

        # Export LaTeX
        self.export_latex(desc_stats, "stationary_descriptive_stats_latex.txt", "Descriptive Statistics for Stationary Variables", "tab:desc_stats", index=True)
        self.export_latex(corr_matrix, "stationary_correlation_matrix_latex.txt", "Correlation Matrix for Stationary Variables", "tab:corr_matrix", index=True)
        self.export_latex(corr_rank, "stationary_correlation_rank_to_dollar_latex.txt", "Correlation Ranking with Dollar Log Return", "tab:corr_rank", index=False)