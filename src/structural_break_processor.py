#!/usr/bin/env python3
"""
Structural Break Processor Module

This module contains the StructuralBreakProcessor class for running
supplementary Chow structural break tests on the stationary monthly dataset.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f


class StructuralBreakProcessor:
    """
    A class for running supplementary Chow tests at thesis regime boundaries.

    These tests are appendix-level robustness checks. They are meant to support
    the regime-based analysis already used in the thesis, not to replace the
    existing interaction models or the pre-defined subperiod regressions.
    """

    def __init__(self, stationary_file: Path, tables_dir: Path):
        """
        Initialize the processor.

        Args:
            stationary_file: Path to the stationary CSV file.
            tables_dir: Directory for table outputs.
        """
        self.stationary_file = stationary_file
        self.tables_dir = tables_dir
        self.data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.search_results: Optional[pd.DataFrame] = None
        self.date_column: Optional[str] = None

    def _get_model_specs(self) -> List[List[str]]:
        """
        Return the common model specifications used in the appendix tests.
        """
        return [
            ['vix_diff'],
            ['vxy_g7_diff'],
            ['vxy_em_diff'],
            ['STLFSI4_diff'],
            ['gepu_diff'],
            ['vix_diff', 'STLFSI4_diff'],
            ['vxy_em_diff', 'STLFSI4_diff'],
        ]

    def load_data(self) -> None:
        """
        Load the stationary dataset, resolve the date column, parse dates, and sort.
        """
        if not self.stationary_file.exists():
            raise FileNotFoundError(f"Stationary file {self.stationary_file} does not exist.")

        self.data = pd.read_csv(self.stationary_file)
        self.date_column = self._resolve_date_column(self.data)
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column], errors='coerce')

        if self.data[self.date_column].isna().any():
            raise ValueError(
                f"Could not parse all values in date column '{self.date_column}' from "
                f"{self.stationary_file}."
            )

        self.data = self.data.sort_values(self.date_column).reset_index(drop=True)

    def _resolve_date_column(self, df: pd.DataFrame) -> str:
        """
        Resolve the date column name from the stationary dataset.
        """
        for candidate in ['Date', 'date']:
            if candidate in df.columns:
                return candidate

        for column in df.columns:
            if column.lower() == 'date':
                return column

        raise ValueError(
            "Missing date column in stationary dataset. Expected one of: "
            "'Date', 'date', or another column named 'date' case-insensitively."
        )

    def validate_columns(self, required_columns: Sequence[str]) -> None:
        """
        Validate that all required columns exist in the loaded data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        missing = [column for column in required_columns if column not in self.data.columns]
        if missing:
            raise ValueError(
                "Missing required columns for structural break tests: "
                + ", ".join(missing)
            )

    def _compute_rss(self, df: pd.DataFrame, dep_var: str, indep_vars: Sequence[str]) -> float:
        """
        Fit a simple OLS model with an intercept and return its RSS.
        """
        X = df[list(indep_vars)].to_numpy(dtype=float)
        intercept = np.ones((len(df), 1), dtype=float)
        design_matrix = np.hstack([intercept, X])
        y = df[dep_var].to_numpy(dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
        residuals = y - design_matrix @ coefficients
        return float(np.sum(np.square(residuals)))

    def _insufficient_result(
        self,
        dep_var: str,
        indep_vars: Sequence[str],
        breakpoint_date: pd.Timestamp,
        n_pre: int,
        n_post: int,
    ) -> Dict[str, object]:
        """
        Build a standardized result row for cases with insufficient data.
        """
        return {
            'model_name': f"{dep_var} ~ {' + '.join(indep_vars)}",
            'dependent_variable': dep_var,
            'independent_variables': ', '.join(indep_vars),
            'breakpoint': breakpoint_date.strftime('%Y-%m-%d'),
            'n_pre': n_pre,
            'n_post': n_post,
            'chow_statistic': np.nan,
            'p_value': np.nan,
            'break_evidence': 'Insufficient data',
            'interpretation': 'Insufficient data to estimate the Chow test reliably',
        }

    def _insufficient_search_result(
        self,
        dep_var: str,
        indep_vars: Sequence[str],
    ) -> Dict[str, object]:
        """
        Build a standardized search result row for insufficient-data cases.
        """
        return {
            'model_name': f"{dep_var} ~ {' + '.join(indep_vars)}",
            'dependent_variable': dep_var,
            'independent_variables': ', '.join(indep_vars),
            'best_breakpoint': '',
            'n_pre': np.nan,
            'n_post': np.nan,
            'max_chow_statistic': np.nan,
            'p_value': np.nan,
            'break_evidence': 'Insufficient data',
            'interpretation': 'Insufficient data to estimate the Chow test reliably',
        }

    def _classify_break_evidence(self, p_value: float) -> Dict[str, str]:
        """
        Translate a p-value into thesis-facing evidence labels.
        """
        if p_value < 0.05:
            return {
                'break_evidence': 'Yes',
                'interpretation': 'Evidence of parameter instability at the selected regime boundary',
            }
        if p_value < 0.10:
            return {
                'break_evidence': 'Weak',
                'interpretation': 'Weak evidence of parameter instability',
            }
        return {
            'break_evidence': 'No',
            'interpretation': 'No evidence of parameter instability at the selected regime boundary',
        }

    def run_chow_test(
        self,
        dataframe: pd.DataFrame,
        dependent_variable: str,
        independent_variables: Sequence[str],
        breakpoint_date: str,
        date_column: str,
    ) -> Dict[str, object]:
        """
        Run one Chow test at a fixed, thesis-defined regime boundary.

        The breakpoint dates correspond to regime boundaries already used in the
        thesis. These tests support the regime-based analysis and should not be
        interpreted as an endogenous search for unknown structural breaks.
        """
        required_columns = [date_column, dependent_variable] + list(independent_variables)
        missing = [column for column in required_columns if column not in dataframe.columns]
        if missing:
            raise ValueError(
                "Missing required columns for Chow test: " + ", ".join(missing)
            )

        breakpoint_ts = pd.Timestamp(breakpoint_date)
        df = dataframe[required_columns].dropna().copy()
        df = df.sort_values(date_column).reset_index(drop=True)

        # The breakpoint observation itself belongs to the post-break sample.
        pre_break = df[df[date_column] < breakpoint_ts].copy()
        post_break = df[df[date_column] >= breakpoint_ts].copy()

        n_pre = len(pre_break)
        n_post = len(post_break)
        k = len(independent_variables) + 1
        denominator_df = n_pre + n_post - (2 * k)

        if n_pre <= k or n_post <= k or denominator_df <= 0:
            return self._insufficient_result(
                dependent_variable,
                independent_variables,
                breakpoint_ts,
                n_pre,
                n_post,
            )

        try:
            rss_pooled = self._compute_rss(df, dependent_variable, independent_variables)
            rss_pre = self._compute_rss(pre_break, dependent_variable, independent_variables)
            rss_post = self._compute_rss(post_break, dependent_variable, independent_variables)
        except Exception:
            return self._insufficient_result(
                dependent_variable,
                independent_variables,
                breakpoint_ts,
                n_pre,
                n_post,
            )

        rss_split = rss_pre + rss_post

        if rss_split <= 0:
            return self._insufficient_result(
                dependent_variable,
                independent_variables,
                breakpoint_ts,
                n_pre,
                n_post,
            )

        numerator = (rss_pooled - rss_split) / k
        denominator = rss_split / denominator_df
        chow_statistic = max(numerator / denominator, 0.0)
        p_value = float(f.sf(chow_statistic, k, denominator_df))
        evidence = self._classify_break_evidence(p_value)

        return {
            'model_name': f"{dependent_variable} ~ {' + '.join(independent_variables)}",
            'dependent_variable': dependent_variable,
            'independent_variables': ', '.join(independent_variables),
            'breakpoint': breakpoint_ts.strftime('%Y-%m-%d'),
            'n_pre': n_pre,
            'n_post': n_post,
            'chow_statistic': chow_statistic,
            'p_value': p_value,
            'break_evidence': evidence['break_evidence'],
            'interpretation': evidence['interpretation'],
        }

    def run_all_structural_break_tests(self) -> None:
        """
        Run all pre-specified structural break tests for appendix robustness.
        """
        if self.data is None or self.date_column is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        dependent_variable = 'dollar_log_return'
        model_specs = self._get_model_specs()
        breakpoints = [
            '2008-01-01',
            '2010-01-01',
            '2020-01-01',
            '2022-01-01',
        ]

        required_columns = {
            self.date_column,
            dependent_variable,
            'vix_diff',
            'vxy_g7_diff',
            'vxy_em_diff',
            'STLFSI4_diff',
            'gepu_diff',
        }
        self.validate_columns(sorted(required_columns))

        results: List[Dict[str, object]] = []
        for spec in model_specs:
            for breakpoint in breakpoints:
                results.append(
                    self.run_chow_test(
                        dataframe=self.data,
                        dependent_variable=dependent_variable,
                        independent_variables=spec,
                        breakpoint_date=breakpoint,
                        date_column=self.date_column,
                    )
                )

        self.results = pd.DataFrame(results)

    def run_breakpoint_search(
        self,
        dataframe: pd.DataFrame,
        dependent_variable: str,
        independent_variables: Sequence[str],
        date_column: str,
        trim_fraction: float = 0.15,
    ) -> Dict[str, object]:
        """
        Search for the strongest single-break Chow test using a trimmed sample.

        This is a simple exploratory breakpoint search based on repeated Chow
        tests. It is supplementary appendix evidence only and should not be
        interpreted as a full Bai-Perron multiple-break procedure.
        """
        required_columns = [date_column, dependent_variable] + list(independent_variables)
        missing = [column for column in required_columns if column not in dataframe.columns]
        if missing:
            raise ValueError(
                "Missing required columns for breakpoint search: " + ", ".join(missing)
            )

        df = dataframe[required_columns].dropna().copy()
        df = df.sort_values(date_column).reset_index(drop=True)
        n_obs = len(df)
        k = len(independent_variables) + 1

        if n_obs == 0:
            return self._insufficient_search_result(dependent_variable, independent_variables)

        trim_count = int(np.ceil(trim_fraction * n_obs))
        min_required = max(trim_count, k + 1)
        candidate_start = min_required
        candidate_stop = n_obs - min_required

        if candidate_start > candidate_stop:
            return self._insufficient_search_result(dependent_variable, independent_variables)

        best_result: Optional[Dict[str, object]] = None

        for idx in range(candidate_start, candidate_stop + 1):
            breakpoint_ts = pd.Timestamp(df.iloc[idx][date_column])
            candidate = self.run_chow_test(
                dataframe=df,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                breakpoint_date=breakpoint_ts.strftime('%Y-%m-%d'),
                date_column=date_column,
            )

            if pd.isna(candidate['chow_statistic']):
                continue

            if best_result is None or candidate['chow_statistic'] > best_result['max_chow_statistic']:
                best_result = {
                    'model_name': candidate['model_name'],
                    'dependent_variable': candidate['dependent_variable'],
                    'independent_variables': candidate['independent_variables'],
                    'best_breakpoint': candidate['breakpoint'],
                    'n_pre': candidate['n_pre'],
                    'n_post': candidate['n_post'],
                    'max_chow_statistic': candidate['chow_statistic'],
                    'p_value': candidate['p_value'],
                    'break_evidence': candidate['break_evidence'],
                    'interpretation': candidate['interpretation'],
                }

        if best_result is None:
            return self._insufficient_search_result(dependent_variable, independent_variables)

        return best_result

    def run_all_breakpoint_searches(self) -> None:
        """
        Run the exploratory data-driven breakpoint search for all model specs.
        """
        if self.data is None or self.date_column is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        dependent_variable = 'dollar_log_return'
        model_specs = self._get_model_specs()
        required_columns = {
            self.date_column,
            dependent_variable,
            'vix_diff',
            'vxy_g7_diff',
            'vxy_em_diff',
            'STLFSI4_diff',
            'gepu_diff',
        }
        self.validate_columns(sorted(required_columns))

        search_results: List[Dict[str, object]] = []
        for spec in model_specs:
            search_results.append(
                self.run_breakpoint_search(
                    dataframe=self.data,
                    dependent_variable=dependent_variable,
                    independent_variables=spec,
                    date_column=self.date_column,
                    trim_fraction=0.15,
                )
            )

        self.search_results = pd.DataFrame(search_results)

    def export_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export results to CSV in tables_dir.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename
        df.to_csv(file_path, index=False)
        print(f"Exported {filename} to {file_path}")

    def export_latex(
        self,
        df: pd.DataFrame,
        filename: str,
        caption: str,
        label: str,
        note: str = '',
    ) -> None:
        """
        Export a compact appendix-ready LaTeX table.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename
        latex_str = self._df_to_latex(df, caption, label, note)
        file_path.write_text(latex_str + '\n')
        print(f"Exported {filename} to {file_path}")

    def _df_to_latex(self, df: pd.DataFrame, caption: str, label: str, note: str = '') -> str:
        """
        Convert a DataFrame to a compact LaTeX table string.
        """
        display_df = df.copy()

        for column in ['chow_statistic', 'max_chow_statistic', 'p_value']:
            if column in display_df.columns:
                display_df[column] = display_df[column].map(
                    lambda value: '' if pd.isna(value) else f"{value:.3f}"
                )

        for column in ['n_pre', 'n_post']:
            if column in display_df.columns:
                display_df[column] = display_df[column].map(
                    lambda value: '' if pd.isna(value) else str(int(value))
                )

        column_format = 'l' * len(display_df.columns)
        lines = [
            '\\begin{table}[H]',
            '\\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            '\\resizebox{\\textwidth}{!}{%',
            f'\\begin{{tabular}}{{{column_format}}}',
            '\\hline',
            ' & '.join(self._escape_latex(str(column)) for column in display_df.columns) + ' \\\\',
            '\\hline',
        ]

        for _, row in display_df.iterrows():
            formatted_row = [self._escape_latex(str(value)) for value in row.values]
            lines.append(' & '.join(formatted_row) + ' \\\\')

        lines.extend([
            '\\hline',
            '\\end{tabular}%',
            '}',
        ])
        if note:
            lines.extend([
                f'\\vspace{{0.3em}}\\parbox{{0.95\\textwidth}}{{\\footnotesize {self._escape_latex(note)}}}',
            ])
        lines.extend([
            '\\end{table}',
        ])
        return '\n'.join(lines)

    def _escape_latex(self, value: str) -> str:
        """
        Escape LaTeX-special characters in cell text.
        """
        replacements = {
            '\\': '\\textbackslash{}',
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}',
        }
        escaped = value
        for old, new in replacements.items():
            escaped = escaped.replace(old, new)
        return escaped

    def run_full_analysis(self) -> None:
        """
        Run the full supplementary structural break workflow and export outputs.
        """
        self.load_data()
        self.run_all_structural_break_tests()
        self.run_all_breakpoint_searches()

        if self.results is None:
            raise ValueError("Structural break results were not generated.")
        if self.search_results is None:
            raise ValueError("Structural break search results were not generated.")

        self.export_csv(self.results, 'structural_break_tests.csv')
        self.export_latex(
            self.results,
            'structural_break_tests_latex.txt',
            'Structural break tests at selected regime boundaries',
            'tab:structural_break_tests',
        )
        self.export_csv(self.search_results, 'structural_break_search.csv')
        self.export_latex(
            self.search_results,
            'structural_break_search_latex.txt',
            'Data-driven structural break search using repeated Chow tests',
            'tab:structural_break_search',
            (
                'Note: This is a simple data-driven breakpoint search based on repeated '
                'Chow tests with 15 percent trimming on each side. It is exploratory '
                'robustness evidence only and not a full Bai-Perron multiple-break procedure.'
            ),
        )
