#!/usr/bin/env python3
"""
Regression Processor Module

This module contains the RegressionProcessor class for performing bivariate
OLS regressions on the stationary dataset with Newey-West HAC standard errors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm
from typing import List, Dict, Optional


class RegressionProcessor:
    """
    A class for running bivariate OLS regressions on stationary data.

    This class loads the stationary dataset, runs regressions for each explanatory
    variable against dollar_log_return, computes rankings, and exports results.
    """

    def __init__(self, stationary_file: Path, tables_dir: Path, level_file: Optional[Path] = None):
        """
        Initialize the processor.

        Args:
            stationary_file: Path to the stationary CSV file.
            tables_dir: Directory for table outputs.
            level_file: Optional path to the level-form monthly CSV file.
        """
        self.stationary_file = stationary_file
        self.tables_dir = tables_dir
        self.level_file = level_file
        self.data: Optional[pd.DataFrame] = None
        self.regression_results: List[Dict] = []
        self.sq2_results: List[Dict] = []
        self.level_robustness_bivariate_results: List[Dict] = []
        self.level_robustness_augmented_results: List[Dict] = []
        self.interaction_results: List[Dict] = []
        self.interaction_augmented_results: List[Dict] = []
        self.subperiod_results: List[Dict] = []
        self.subperiod_augmented_results: List[Dict] = []
        self.sq3_summary: Optional[pd.DataFrame] = None
        self.ranking: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """
        Load the stationary dataset, parse dates, and sort.
        """
        if not self.stationary_file.exists():
            raise FileNotFoundError(f"Stationary file {self.stationary_file} does not exist.")
        self.data = pd.read_csv(self.stationary_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)

    def load_level_robustness_data(self) -> None:
        """
        Load and merge level and transformed series for appendix robustness tests.

        This appendix-level check keeps the main dependent variable as
        dollar_log_return because the dollar index is non-stationary in levels.
        It uses level specifications only for explanatory variables that are
        stationary in levels, while GEPU remains as gepu_diff because GEPU is
        non-stationary in levels. The goal is to assess sensitivity to the
        first-difference choice without changing the main thesis design.
        """
        if not self.stationary_file.exists():
            raise FileNotFoundError(f"Stationary file {self.stationary_file} does not exist.")

        if self.level_file is None:
            self.level_file = self.stationary_file.parent / "master_monthly.csv"

        if not self.level_file.exists():
            raise FileNotFoundError(f"Level file {self.level_file} does not exist.")

        stationary_df = pd.read_csv(self.stationary_file)
        level_df = pd.read_csv(self.level_file)

        stationary_df['Date'] = pd.to_datetime(stationary_df['Date'])
        level_df['Date'] = pd.to_datetime(level_df['Date'])

        stationary_df = stationary_df.sort_values('Date').reset_index(drop=True)
        level_df = level_df.sort_values('Date').reset_index(drop=True)

        # Keep the short-run dependent variable and GEPU difference from the
        # stationary file, but use level-form regressors for variables that are
        # stationary in levels.
        stationary_subset = stationary_df[['Date', 'dollar_log_return', 'gepu_diff']]
        level_subset = level_df[['Date', 'vix', 'STLFSI4', 'vxy_g7', 'vxy_em']]

        self.data = pd.merge(stationary_subset, level_subset, on='Date', how='inner')
        self.data = self.data.sort_values('Date').reset_index(drop=True)

        required_columns = [
            'Date',
            'dollar_log_return',
            'vix',
            'STLFSI4',
            'vxy_g7',
            'vxy_em',
            'gepu_diff',
        ]
        missing = [column for column in required_columns if column not in self.data.columns]
        if missing:
            raise ValueError(
                "Missing required columns for level robustness regressions: "
                + ", ".join(missing)
            )

    def run_bivariate_regression(self, dep_var: str, exp_var: str) -> Dict:
        """
        Run a single bivariate OLS regression with Newey-West HAC SE.

        Args:
            dep_var: Dependent variable name.
            exp_var: Explanatory variable name.

        Returns:
            Dict with regression results.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        df = self.data[[dep_var, exp_var]].dropna()
        if len(df) < 10:  # Arbitrary minimum obs
            return {
                'dependent_variable': dep_var,
                'explanatory_variable': exp_var,
                'coefficient': np.nan,
                'standard_error': np.nan,
                't_statistic': np.nan,
                'p_value': np.nan,
                'r_squared': np.nan,
                'adjusted_r_squared': np.nan,
                'n_obs': len(df)
            }
        X = add_constant(df[exp_var])
        y = df[dep_var]
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        return {
            'dependent_variable': dep_var,
            'explanatory_variable': exp_var,
            'coefficient': model.params[exp_var],
            'standard_error': model.bse[exp_var],
            't_statistic': model.tvalues[exp_var],
            'p_value': model.pvalues[exp_var],
            'r_squared': model.rsquared,
            'adjusted_r_squared': model.rsquared_adj,
            'n_obs': model.nobs
        }

    def run_all_bivariate_regressions(self) -> None:
        """
        Run bivariate regressions for all explanatory variables.
        """
        dep_var = 'dollar_log_return'
        exp_vars = ['vix_diff', 'STLFSI4_diff', 'vxy_g7_diff', 'vxy_em_diff', 'gepu_diff']
        self.regression_results = []
        for exp_var in exp_vars:
            result = self.run_bivariate_regression(dep_var, exp_var)
            self.regression_results.append(result)

    def build_ranking(self) -> None:
        """
        Build a ranking DataFrame based on p-value, adj_r2, and abs_corr.
        """
        if not self.regression_results:
            raise ValueError("Regression results not available. Call run_all_bivariate_regressions() first.")
        df_results = pd.DataFrame(self.regression_results)
        # Get correlations
        corr_series = self.data[['dollar_log_return', 'vix_diff', 'STLFSI4_diff', 'vxy_g7_diff', 'vxy_em_diff', 'gepu_diff']].corr()['dollar_log_return']
        df_results['abs_corr'] = df_results['explanatory_variable'].map(lambda x: abs(corr_series[x]))
        # Rank: lower p_value, higher adj_r2, higher abs_corr
        df_results['rank_score'] = df_results['p_value'] - df_results['adjusted_r_squared'] - df_results['abs_corr']
        self.ranking = df_results.sort_values('rank_score').reset_index(drop=True)[['explanatory_variable', 'p_value', 'adjusted_r_squared', 'abs_corr']]

    def export_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export DataFrame to CSV in tables_dir.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename
        df.to_csv(file_path, index=False)
        print(f"Exported {filename} to {file_path}")

    def export_latex(self, df: pd.DataFrame, filename: str, caption: str = "", label: str = "") -> None:
        """
        Export DataFrame to LaTeX table format, rounded to 3 decimals.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename
        df_rounded = df.round(3)
        latex_str = self._df_to_latex(df_rounded, caption, label)
        with open(file_path, 'w') as f:
            f.write(latex_str)
        print(f"Exported {filename} to {file_path}")

    def _df_to_latex(self, df: pd.DataFrame, caption: str, label: str) -> str:
        """
        Convert DataFrame to LaTeX table string manually.
        """
        lines = []
        num_cols = len(df.columns)
        if caption:
            lines.append("\\begin{table}[H]")
            lines.append("\\centering")
            lines.append(f"\\caption{{{caption}}}")
            if label:
                lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{" + "c" * num_cols + "}")
        lines.append("\\hline")
        header = " & ".join(df.columns)
        lines.append(header + " \\\\")
        lines.append("\\hline")
        for _, row in df.iterrows():
            formatted_values = []
            for val in row.values:
                if isinstance(val, (int, float)) and not np.isnan(val):
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

    def _df_to_latex_resized(self, df: pd.DataFrame, caption: str, label: str) -> str:
        """
        Convert DataFrame to a compact LaTeX table string with resizebox.
        """
        df_rounded = df.round(3)
        lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}"]
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append("\\begin{tabular}{" + "c" * len(df_rounded.columns) + "}")
        lines.append("\\hline")
        lines.append(" & ".join(df_rounded.columns) + " \\\\")
        lines.append("\\hline")
        for _, row in df_rounded.iterrows():
            formatted_values = []
            for col_name in df_rounded.columns:
                val = row[col_name]
                if col_name == 'n_obs' and isinstance(val, (int, float, np.integer, np.floating)) and not np.isnan(val):
                    formatted_values.append(str(int(val)))
                    continue
                if isinstance(val, (int, float, np.integer, np.floating)) and not np.isnan(val):
                    formatted_values.append(f"{val:.3f}")
                else:
                    formatted_values.append(str(val))
            lines.append(" & ".join(formatted_values) + " \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    def run_full_analysis(self) -> None:
        """
        Run the full bivariate regression analysis: load, regress, rank, export.
        """
        self.load_data()
        self.run_all_bivariate_regressions()
        self.build_ranking()
        # Export CSVs
        df_results = pd.DataFrame(self.regression_results)
        self.export_csv(df_results, "bivariate_regressions_stationary.csv")
        self.export_csv(self.ranking, "bivariate_regression_ranking.csv")
        # Export LaTeX
        self.export_latex(df_results, "bivariate_regressions_stationary_latex.txt", "Bivariate Regression Results", "tab:bivar_reg")
        self.export_latex(self.ranking, "bivariate_regression_ranking_latex.txt", "Regression Ranking", "tab:reg_rank")

    def run_multivariate_regression(self, dep_var: str, exp_vars: List[str]) -> Dict:
        """
        Run a single multivariate OLS regression with Newey-West HAC SE.

        Args:
            dep_var: Dependent variable name.
            exp_vars: List of explanatory variable names.

        Returns:
            Dict with regression results.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        df = self.data[[dep_var] + exp_vars].dropna()
        if len(df) < 10:
            result = {
                'model_name': f"{dep_var} ~ {' + '.join(exp_vars)}",
                'dependent_variable': dep_var,
                'explanatory_variables': ' + '.join(exp_vars),
                'coefficient_var1': np.nan,
                'std_error_var1': np.nan,
                't_stat_var1': np.nan,
                'p_value_var1': np.nan,
                'coefficient_var2': np.nan,
                'std_error_var2': np.nan,
                't_stat_var2': np.nan,
                'p_value_var2': np.nan,
                'r_squared': np.nan,
                'adjusted_r_squared': np.nan,
                'n_obs': len(df)
            }
            return result
        X = add_constant(df[exp_vars])
        y = df[dep_var]
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        return {
            'model_name': f"{dep_var} ~ {' + '.join(exp_vars)}",
            'dependent_variable': dep_var,
            'explanatory_variables': ' + '.join(exp_vars),
            'coefficient_var1': model.params[exp_vars[0]],
            'std_error_var1': model.bse[exp_vars[0]],
            't_stat_var1': model.tvalues[exp_vars[0]],
            'p_value_var1': model.pvalues[exp_vars[0]],
            'coefficient_var2': model.params[exp_vars[1]],
            'std_error_var2': model.bse[exp_vars[1]],
            't_stat_var2': model.tvalues[exp_vars[1]],
            'p_value_var2': model.pvalues[exp_vars[1]],
            'r_squared': model.rsquared,
            'adjusted_r_squared': model.rsquared_adj,
            'n_obs': model.nobs
        }

    def run_level_robustness_bivariate_regressions(self) -> None:
        """
        Run appendix-only bivariate regressions with level-form stationary regressors.
        """
        dep_var = 'dollar_log_return'
        exp_vars = ['vix', 'STLFSI4', 'vxy_g7', 'vxy_em', 'gepu_diff']
        self.level_robustness_bivariate_results = []

        for exp_var in exp_vars:
            base_result = self.run_bivariate_regression(dep_var, exp_var)
            self.level_robustness_bivariate_results.append({
                'model_type': 'bivariate',
                'model_name': f"{dep_var} ~ {exp_var}",
                'dependent_variable': dep_var,
                'independent_variable': exp_var,
                'coefficient': base_result['coefficient'],
                'standard_error': base_result['standard_error'],
                't_statistic': base_result['t_statistic'],
                'p_value': base_result['p_value'],
                'r_squared': base_result['r_squared'],
                'adjusted_r_squared': base_result['adjusted_r_squared'],
                'n_obs': base_result['n_obs'],
                'proxy_variable': np.nan,
                'proxy_coefficient': np.nan,
                'proxy_standard_error': np.nan,
                'proxy_t_statistic': np.nan,
                'proxy_p_value': np.nan,
                'added_variable': np.nan,
                'added_variable_coefficient': np.nan,
                'added_variable_standard_error': np.nan,
                'added_variable_t_statistic': np.nan,
                'added_variable_p_value': np.nan,
            })

    def run_level_robustness_augmented_regressions(self) -> None:
        """
        Run appendix-only augmented regressions with mixed level/difference inputs.
        """
        dep_var = 'dollar_log_return'
        model_specs = [
            ['vix', 'STLFSI4'],
            ['vix', 'gepu_diff'],
            ['vxy_g7', 'STLFSI4'],
            ['vxy_g7', 'gepu_diff'],
            ['vxy_em', 'STLFSI4'],
            ['vxy_em', 'gepu_diff'],
        ]
        self.level_robustness_augmented_results = []

        for exp_vars in model_specs:
            base_result = self.run_multivariate_regression(dep_var, exp_vars)
            self.level_robustness_augmented_results.append({
                'model_type': 'augmented',
                'model_name': base_result['model_name'],
                'dependent_variable': dep_var,
                'independent_variable': np.nan,
                'coefficient': np.nan,
                'standard_error': np.nan,
                't_statistic': np.nan,
                'p_value': np.nan,
                'proxy_variable': exp_vars[0],
                'proxy_coefficient': base_result['coefficient_var1'],
                'proxy_standard_error': base_result['std_error_var1'],
                'proxy_t_statistic': base_result['t_stat_var1'],
                'proxy_p_value': base_result['p_value_var1'],
                'added_variable': exp_vars[1],
                'added_variable_coefficient': base_result['coefficient_var2'],
                'added_variable_standard_error': base_result['std_error_var2'],
                'added_variable_t_statistic': base_result['t_stat_var2'],
                'added_variable_p_value': base_result['p_value_var2'],
                'r_squared': base_result['r_squared'],
                'adjusted_r_squared': base_result['adjusted_r_squared'],
                'n_obs': base_result['n_obs'],
            })

    def export_level_robustness_latex(
        self,
        bivariate_df: pd.DataFrame,
        augmented_df: pd.DataFrame,
        filename: str,
    ) -> None:
        """
        Export appendix-ready LaTeX tables for the level-based robustness check.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename

        bivariate_table = self._df_to_latex_resized(
            bivariate_df,
            "Level-based bivariate robustness regressions",
            "tab:level_robustness_bivariate",
        )
        augmented_table = self._df_to_latex_resized(
            augmented_df,
            "Level-based augmented robustness regressions",
            "tab:level_robustness_augmented",
        )

        note = (
            "% Appendix robustness check only. Main thesis specifications use "
            "first differences for consistency with the short-run change-based design. "
            "Level specifications are estimated only for regressors found to be "
            "stationary in levels, while GEPU remains as gepu_diff."
        )
        with open(file_path, 'w') as f:
            f.write(note + "\n\n" + bivariate_table + "\n\n" + augmented_table + "\n")
        print(f"Exported {filename} to {file_path}")

    def run_level_robustness_analysis(self) -> None:
        """
        Run appendix-only robustness regressions using levels where justified.

        The main thesis still relies on first differences for consistency with
        the short-run empirical design. This robustness check estimates level
        specifications only for explanatory variables that are stationary in
        levels, keeps GEPU as gepu_diff because it is non-stationary in levels,
        and keeps the dependent variable as dollar_log_return because the dollar
        index is non-stationary in levels.
        """
        self.load_level_robustness_data()
        self.run_level_robustness_bivariate_regressions()
        self.run_level_robustness_augmented_regressions()

        bivariate_df = pd.DataFrame(self.level_robustness_bivariate_results)
        augmented_df = pd.DataFrame(self.level_robustness_augmented_results)
        if not bivariate_df.empty:
            bivariate_df['n_obs'] = bivariate_df['n_obs'].astype(int)
        if not augmented_df.empty:
            augmented_df['n_obs'] = augmented_df['n_obs'].astype(int)
        combined_df = pd.concat([bivariate_df, augmented_df], ignore_index=True, sort=False)
        if not combined_df.empty:
            combined_df['n_obs'] = combined_df['n_obs'].astype(int)

        self.export_csv(combined_df, "level_robustness_regressions.csv")

        bivariate_latex_df = bivariate_df[
            [
                'model_name',
                'independent_variable',
                'coefficient',
                'standard_error',
                't_statistic',
                'p_value',
                'adjusted_r_squared',
                'n_obs',
            ]
        ]
        augmented_latex_df = augmented_df[
            [
                'model_name',
                'proxy_variable',
                'proxy_coefficient',
                'proxy_standard_error',
                'proxy_t_statistic',
                'proxy_p_value',
                'added_variable',
                'added_variable_coefficient',
                'added_variable_standard_error',
                'added_variable_t_statistic',
                'added_variable_p_value',
                'adjusted_r_squared',
                'n_obs',
            ]
        ]
        self.export_level_robustness_latex(
            bivariate_latex_df,
            augmented_latex_df,
            "level_robustness_regressions_latex.txt",
        )

    def run_sq2_multivariate_regressions(self) -> None:
        """
        Run the SQ2 multivariate regression specifications.
        """
        dep_var = 'dollar_log_return'
        model_specs = [
            ('dollar_log_return ~ vix_diff + STLFSI4_diff', ['vix_diff', 'STLFSI4_diff']),
            ('dollar_log_return ~ vix_diff + gepu_diff', ['vix_diff', 'gepu_diff']),
            ('dollar_log_return ~ vxy_g7_diff + STLFSI4_diff', ['vxy_g7_diff', 'STLFSI4_diff']),
            ('dollar_log_return ~ vxy_g7_diff + gepu_diff', ['vxy_g7_diff', 'gepu_diff']),
            ('dollar_log_return ~ vxy_em_diff + STLFSI4_diff', ['vxy_em_diff', 'STLFSI4_diff']),
            ('dollar_log_return ~ vxy_em_diff + gepu_diff', ['vxy_em_diff', 'gepu_diff']),
        ]
        self.sq2_results = []
        for model_name, exp_vars in model_specs:
            result = self.run_multivariate_regression(dep_var, exp_vars)
            self.sq2_results.append(result)

    def build_sq2_summary(self) -> pd.DataFrame:
        """
        Build a compact summary for SQ2 multivariate results.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if not hasattr(self, 'sq2_results') or not self.sq2_results:
            raise ValueError("SQ2 multivariate results not available. Call run_sq2_multivariate_regressions() first.")

        bivariate_map = {}
        if self.regression_results:
            for row in self.regression_results:
                if row['explanatory_variable'] in ['vix_diff', 'vxy_g7_diff', 'vxy_em_diff']:
                    bivariate_map[row['explanatory_variable']] = row['adjusted_r_squared']

        summary_rows = []
        for row in self.sq2_results:
            exp_vars = row['explanatory_variables'].split(' + ')
            volatility_proxy = exp_vars[0]
            added_variable = exp_vars[1]
            added_significant = 'yes' if row['p_value_var2'] < 0.05 else 'no'
            base_adj_r2 = bivariate_map.get(volatility_proxy, np.nan)
            improves = 'yes' if not np.isnan(base_adj_r2) and row['adjusted_r_squared'] > base_adj_r2 else 'no'
            conclusion = 'added variable contributes' if (added_significant == 'yes' and improves == 'yes') else 'added variable does not contribute clearly'
            summary_rows.append({
                'model_name': row['model_name'],
                'volatility_proxy': volatility_proxy,
                'added_variable': added_variable,
                'added_significant': added_significant,
                'adjusted_r_squared_multivariate': row['adjusted_r_squared'],
                'adjusted_r_squared_bivariate': base_adj_r2,
                'adj_r_squared_improves': improves,
                'conclusion': conclusion
            })

        return pd.DataFrame(summary_rows)

    def run_sq2_analysis(self) -> None:
        """
        Run the full SQ2 analysis: load data, run bivariate and multivariate regressions,
        build summary, and export results.
        """
        self.load_data()
        self.run_all_bivariate_regressions()
        self.build_ranking()
        self.run_sq2_multivariate_regressions()
        summary = self.build_sq2_summary()
        df_results = pd.DataFrame(self.sq2_results)
        self.export_csv(df_results, "sq2_multivariate_regressions.csv")
        self.export_csv(summary, "sq2_multivariate_regression_summary.csv")
        self.export_latex(df_results, "sq2_multivariate_regressions_latex.txt", "SQ2 Multivariate Regression Results", "tab:sq2_multivar")
        self.export_latex(summary, "sq2_multivariate_regression_summary_latex.txt", "SQ2 Regression Summary", "tab:sq2_summary")

    def create_crisis_dummies(self) -> None:
        """
        Add crisis dummy variables for the SQ3 interaction models.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.data['GFC_dummy'] = ((self.data['Date'] >= pd.Timestamp('2008-07-01')) &
                                 (self.data['Date'] <= pd.Timestamp('2009-06-01'))).astype(int)
        self.data['COVID_dummy'] = ((self.data['Date'] >= pd.Timestamp('2020-01-01')) &
                                   (self.data['Date'] <= pd.Timestamp('2020-06-01'))).astype(int)

    def _format_interaction_model_name(self, proxy_var: str, dummy_var: str, control_var: Optional[str] = None) -> str:
        """
        Build a compact thesis-style interaction model label.
        """
        proxy_labels = {
            'vix_diff': 'ΔVIX',
            'vxy_g7_diff': 'ΔVXY_G7',
            'vxy_em_diff': 'ΔVXY_EM',
        }
        control_labels = {
            'STLFSI4_diff': 'ΔSTLFSI4',
            'gepu_diff': 'ΔGEPU',
        }
        crisis_labels = {
            'GFC_dummy': 'GFC',
            'COVID_dummy': 'COVID',
        }

        model_name = proxy_labels.get(proxy_var, proxy_var)
        if control_var is not None:
            model_name += f" + {control_labels.get(control_var, control_var)}"
        return f"{model_name} × {crisis_labels.get(dummy_var, dummy_var)}"

    def run_interaction_model(self, dep_var: str, proxy_var: str, dummy_var: str, control_var: Optional[str] = None) -> Dict:
        """
        Run one interaction regression model with an optional additional control.

        Args:
            dep_var: Dependent variable name.
            proxy_var: Volatility proxy variable name.
            dummy_var: Crisis dummy variable name.
            control_var: Optional control variable such as STLFSI4_diff or gepu_diff.

        Returns:
            Dict with regression results.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        variables = [dep_var, proxy_var, dummy_var]
        if control_var is not None:
            variables.append(control_var)
        df = self.data[variables].dropna()
        model_name = self._format_interaction_model_name(proxy_var, dummy_var, control_var)
        explanatory_variables = [proxy_var]
        if control_var is not None:
            explanatory_variables.append(control_var)
        explanatory_variables.extend([dummy_var, f'{proxy_var}:{dummy_var}'])

        if len(df) < 10:
            return {
                'model_name': model_name,
                'dependent_variable': dep_var,
                'explanatory_variables': ' + '.join(explanatory_variables),
                'coefficient_main_proxy': np.nan,
                'p_value_main_proxy': np.nan,
                'coefficient_crisis_dummy': np.nan,
                'p_value_crisis_dummy': np.nan,
                'coefficient_interaction': np.nan,
                'p_value_interaction': np.nan,
                'r_squared': np.nan,
                'adjusted_r_squared': np.nan,
                'n_obs': len(df)
            }

        df['interaction'] = df[proxy_var] * df[dummy_var]
        X_vars = [proxy_var, dummy_var, 'interaction']
        if control_var is not None:
            X_vars.insert(1, control_var)
        X = add_constant(df[X_vars])
        y = df[dep_var]
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

        return {
            'model_name': model_name,
            'dependent_variable': dep_var,
            'explanatory_variables': ' + '.join(explanatory_variables),
            'coefficient_main_proxy': model.params[proxy_var],
            'p_value_main_proxy': model.pvalues[proxy_var],
            'coefficient_crisis_dummy': model.params[dummy_var],
            'p_value_crisis_dummy': model.pvalues[dummy_var],
            'coefficient_interaction': model.params['interaction'],
            'p_value_interaction': model.pvalues['interaction'],
            'r_squared': model.rsquared,
            'adjusted_r_squared': model.rsquared_adj,
            'n_obs': model.nobs
        }

    def run_all_interaction_models(self) -> None:
        """
        Run all SQ3 interaction model specifications.
        """
        dep_var = 'dollar_log_return'
        proxies = ['vix_diff', 'vxy_g7_diff', 'vxy_em_diff']
        self.interaction_results = []
        self.interaction_augmented_results = []

        for proxy in proxies:
            for dummy_var in ['GFC_dummy', 'COVID_dummy']:
                self.interaction_results.append(
                    self.run_interaction_model(dep_var, proxy, dummy_var)
                )
                self.interaction_augmented_results.append(
                    self.run_interaction_model(dep_var, proxy, dummy_var, control_var='STLFSI4_diff')
                )
                self.interaction_augmented_results.append(
                    self.run_interaction_model(dep_var, proxy, dummy_var, control_var='gepu_diff')
                )

    def define_subperiods(self) -> List[Dict[str, object]]:
        """
        Define the SQ3 subperiod regimes.
        """
        return [
            {'name': 'Pre-GFC', 'start': pd.Timestamp('2006-01-01'), 'end': pd.Timestamp('2007-12-01')},
            {'name': 'GFC', 'start': pd.Timestamp('2008-01-01'), 'end': pd.Timestamp('2009-12-01')},
            {'name': 'Post-GFC / pre-COVID', 'start': pd.Timestamp('2010-01-01'), 'end': pd.Timestamp('2019-12-01')},
            {'name': 'COVID period', 'start': pd.Timestamp('2020-01-01'), 'end': pd.Timestamp('2021-12-01')},
            {'name': 'Post-COVID / geopolitical', 'start': pd.Timestamp('2022-01-01'), 'end': pd.Timestamp('2025-12-01')},
        ]

    def _format_subperiod_model_name(self, exp_vars: List[str]) -> str:
        """
        Build a compact thesis-style subperiod model label.
        """
        variable_labels = {
            'vix_diff': 'VIX',
            'vxy_g7_diff': 'VXY-G7',
            'vxy_em_diff': 'VXY-EM',
            'STLFSI4_diff': 'STLFSI4',
            'gepu_diff': 'GEPU',
        }
        return ' + '.join(variable_labels.get(var, var) for var in exp_vars)

    def run_subperiod_regression(self, subperiod_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp, dep_var: str, exp_vars: List[str]) -> Dict:
        """
        Run one subperiod regression model.

        Args:
            subperiod_name: Name of the subperiod.
            start_date: Start date of the subperiod.
            end_date: End date of the subperiod.
            dep_var: Dependent variable name.
            exp_vars: Explanatory variable names.

        Returns:
            Dict with regression results.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        period_df = self.data[(self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)].copy()
        df = period_df[[dep_var] + exp_vars].dropna()
        model_name = self._format_subperiod_model_name(exp_vars)
        added_proxy = exp_vars[1] if len(exp_vars) > 1 else None
        result = {
            'subperiod': subperiod_name,
            'model_name': model_name,
            'dependent_variable': dep_var,
            'explanatory_variables': ' + '.join(exp_vars),
            'coefficient_main_proxy': np.nan,
            'p_value_main_proxy': np.nan,
            'added_proxy': added_proxy,
            'coefficient_added_proxy': np.nan,
            'p_value_added_proxy': np.nan,
            'r_squared': np.nan,
            'adjusted_r_squared': np.nan,
            'n_obs': len(df)
        }
        if len(df) < 10:
            return result
        X = add_constant(df[exp_vars])
        y = df[dep_var]
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        result['coefficient_main_proxy'] = model.params[exp_vars[0]]
        result['p_value_main_proxy'] = model.pvalues[exp_vars[0]]
        if added_proxy is not None:
            result['coefficient_added_proxy'] = model.params[added_proxy]
            result['p_value_added_proxy'] = model.pvalues[added_proxy]
        result['r_squared'] = model.rsquared
        result['adjusted_r_squared'] = model.rsquared_adj
        result['n_obs'] = model.nobs
        return result

    def run_all_subperiod_regressions(self) -> None:
        """
        Run all SQ3 subperiod regression specifications.
        """
        dep_var = 'dollar_log_return'
        self.subperiod_results = []
        self.subperiod_augmented_results = []
        subperiods = self.define_subperiods()
        main_specs = [['vix_diff'], ['vxy_g7_diff'], ['vxy_em_diff']]
        augmented_specs = [
            ['vix_diff', 'STLFSI4_diff'],
            ['vxy_g7_diff', 'STLFSI4_diff'],
            ['vxy_em_diff', 'STLFSI4_diff'],
            ['vix_diff', 'gepu_diff'],
            ['vxy_g7_diff', 'gepu_diff'],
            ['vxy_em_diff', 'gepu_diff'],
        ]

        for period in subperiods:
            for spec in main_specs:
                self.subperiod_results.append(self.run_subperiod_regression(period['name'], period['start'], period['end'], dep_var, spec))
            for spec in augmented_specs:
                self.subperiod_augmented_results.append(self.run_subperiod_regression(period['name'], period['start'], period['end'], dep_var, spec))

    def build_sq3_summary(self) -> pd.DataFrame:
        """
        Build a short interpretation-oriented SQ3 summary table.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if not self.interaction_results:
            raise ValueError("Interaction results not available. Call run_all_interaction_models() first.")
        if not self.subperiod_results:
            raise ValueError("Subperiod results not available. Call run_all_subperiod_regressions() first.")

        rows = []
        proxies = ['vix_diff', 'vxy_g7_diff', 'vxy_em_diff']
        for proxy in proxies:
            proxy_interactions = [row for row in self.interaction_results if proxy in row['explanatory_variables']]
            interaction_significant = any(row['p_value_interaction'] < 0.05 for row in proxy_interactions)
            proxy_subperiods = [row for row in self.subperiod_results if proxy in row['explanatory_variables']]
            proxy_subperiods = sorted(proxy_subperiods, key=lambda x: abs(x['coefficient_main_proxy']) if not np.isnan(x['coefficient_main_proxy']) else -1, reverse=True)
            strongest_subperiods = '; '.join([row['subperiod'] for row in proxy_subperiods[:2] if not np.isnan(row['coefficient_main_proxy'])])
            crisis_coefs = [row for row in proxy_subperiods if row['subperiod'] in ['GFC', 'COVID period']]
            crisis_stronger = any(abs(row['coefficient_main_proxy']) >= max(abs(r['coefficient_main_proxy']) if not np.isnan(r['coefficient_main_proxy']) else 0 for r in proxy_subperiods) for row in crisis_coefs)
            if interaction_significant and crisis_coefs:
                conclusion = 'supports regime dependence'
            elif interaction_significant or crisis_stronger:
                conclusion = 'mixed evidence'
            else:
                conclusion = 'limited formal crisis effect'
            rows.append({
                'proxy': proxy,
                'interaction_significant': 'yes' if interaction_significant else 'no',
                'strongest_subperiods': strongest_subperiods,
                'crisis_regime_evidence': 'yes' if crisis_stronger else 'no',
                'conclusion': conclusion
            })

        self.sq3_summary = pd.DataFrame(rows)
        return self.sq3_summary

    def run_sq3_analysis(self) -> None:
        """
        Run the full SQ3 regime analysis workflow and export results.
        """
        self.load_data()
        self.create_crisis_dummies()
        self.run_all_interaction_models()
        self.run_all_subperiod_regressions()
        summary = self.build_sq3_summary()

        self.export_csv(pd.DataFrame(self.interaction_results), "sq3_interaction_models.csv")
        self.export_csv(pd.DataFrame(self.interaction_augmented_results), "sq3_interaction_models_augmented.csv")
        self.export_csv(pd.DataFrame(self.subperiod_results), "sq3_subperiod_regressions.csv")
        self.export_csv(pd.DataFrame(self.subperiod_augmented_results), "sq3_subperiod_regressions_augmented.csv")
        self.export_csv(summary, "sq3_regime_summary.csv")

        self.export_latex(pd.DataFrame(self.interaction_results), "sq3_interaction_models_latex.txt", "SQ3 Interaction Models", "tab:sq3_interaction")
        self.export_latex(pd.DataFrame(self.interaction_augmented_results), "sq3_interaction_models_augmented_latex.txt", "SQ3 Interaction Models Augmented", "tab:sq3_interaction_aug")
        self.export_latex(pd.DataFrame(self.subperiod_results), "sq3_subperiod_regressions_latex.txt", "SQ3 Subperiod Regressions", "tab:sq3_subperiod")
        self.export_latex(pd.DataFrame(self.subperiod_augmented_results), "sq3_subperiod_regressions_augmented_latex.txt", "SQ3 Subperiod Regressions Augmented", "tab:sq3_subperiod_aug")
        self.export_latex(summary, "sq3_regime_summary_latex.txt", "SQ3 Regime Summary", "tab:sq3_summary")
