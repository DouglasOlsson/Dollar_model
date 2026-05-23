#!/usr/bin/env python3
"""
PCA Processor Module

This module contains PCAProcessor, which performs PCA on stationary uncertainty
variables and runs complementary PC1 regression diagnostics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from typing import List, Dict, Optional


class PCAProcessor:
    """
    A class to run PCA analysis on stationary uncertainty variables.

    This class loads the stationary dataset, performs PCA for three specifications,
    exports explained variance, loadings, scores, correlations, and runs simple
    PC1 regressions.
    """

    def __init__(self, stationary_file: Path, tables_dir: Path, processed_dir: Path):
        """
        Initialize the PCA processor.

        Args:
            stationary_file: Path to the stationary CSV file.
            tables_dir: Directory for table outputs.
            processed_dir: Directory for processed CSV outputs.
        """
        self.stationary_file = stationary_file
        self.tables_dir = tables_dir
        self.processed_dir = processed_dir
        self.data: Optional[pd.DataFrame] = None
        self.pca_results: Dict[str, Dict] = {}
        self.summary: Optional[pd.DataFrame] = None
        self.regression_results: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """
        Load the stationary dataset, parse Date, and sort.
        """
        if not self.stationary_file.exists():
            raise FileNotFoundError(f"Stationary file {self.stationary_file} does not exist.")
        self.data = pd.read_csv(self.stationary_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)

    def _select_variables(self, variables: List[str]) -> pd.DataFrame:
        """
        Select and return the variables to use for PCA, dropping missing rows.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        df = self.data[['Date'] + variables].dropna()
        return df

    def _standardize(self, df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        """
        Standardize variables using z-scores.
        """
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[variables])
        return pd.DataFrame(scaled, columns=variables, index=df.index)

    def run_pca(self, spec_name: str, variables: List[str]) -> None:
        """
        Run PCA for one specification.

        Args:
            spec_name: Name of the specification (e.g. 'A', 'B', or 'C').
            variables: List of variables to include.
        """
        df = self._select_variables(variables)
        standardized = self._standardize(df, variables)

        pca = PCA()
        scores = pca.fit_transform(standardized)
        pc_columns = [f'PC{i+1}' for i in range(scores.shape[1])]

        explained_variance = pd.DataFrame({
            'component': pc_columns,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_)
        })

        loadings = pd.DataFrame(
            pca.components_.T,
            index=variables,
            columns=pc_columns
        )

        component_scores = pd.DataFrame(
            scores,
            columns=pc_columns,
            index=df.index
        )

        combined = pd.concat([standardized, component_scores], axis=1)
        correlations = combined[variables + pc_columns].corr().loc[variables, pc_columns]

        scores_df = pd.DataFrame(
            scores,
            columns=[f'PC{i+1}_{spec_name}' for i in range(scores.shape[1])],
            index=df.index
        )
        scores_df = pd.concat([df[['Date']].reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

        self.pca_results[spec_name] = {
            'variables': variables,
            'explained_variance': explained_variance,
            'loadings': loadings,
            'scores': scores_df,
            'correlations': correlations
        }

    def export_pca_outputs(self, spec_name: str) -> None:
        """
        Export PCA outputs for a given specification, including explained
        variance, loadings, scores, and correlations.
        """
        if spec_name not in self.pca_results:
            raise ValueError(f"PCA results for spec {spec_name} not found.")
        result = self.pca_results[spec_name]
        explained_file = self.tables_dir / f"pca_{spec_name}_explained_variance.csv"
        loadings_file = self.tables_dir / f"pca_{spec_name}_loadings.csv"
        scores_file = self.processed_dir / f"pca_{spec_name}_scores.csv"
        correlations_file = self.tables_dir / f"pca_{spec_name}_correlations.csv"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        result['explained_variance'].to_csv(explained_file, index=False)
        result['loadings'].to_csv(loadings_file)
        result['scores'].to_csv(scores_file, index=False)
        result['correlations'].to_csv(correlations_file)
        self.export_latex(
            result['correlations'].reset_index().rename(columns={'index': 'variable'}),
            f'pca_{spec_name}_correlations_latex.txt',
            f'PCA {spec_name} Correlations',
            f'tab:pca_{spec_name.lower()}_corr'
        )

    def _regression(self, y: pd.Series, X: pd.DataFrame) -> Dict:
        """
        Run OLS with Newey-West HAC standard errors.
        """
        predictor_name = X.columns[0]
        X = add_constant(X)
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        return {
            'model_name': None,
            'coefficient': model.params[predictor_name],
            'standard_error': model.bse[predictor_name],
            't_statistic': model.tvalues[predictor_name],
            'p_value': model.pvalues[predictor_name],
            'r_squared': model.rsquared,
            'adjusted_r_squared': model.rsquared_adj,
            'n_obs': model.nobs
        }

    def run_pca_regressions(self) -> None:
        """
        Run simple regressions using PC1 from each PCA specification.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        rows = []
        for spec_name in ['A', 'B', 'C']:
            score_df = self.pca_results[spec_name]['scores']
            merged = pd.merge(self.data, score_df, on='Date', how='inner')
            merged = merged.dropna(subset=[f'PC1_{spec_name}', 'dollar_log_return'])

            # PC1 regression
            y = merged['dollar_log_return']
            X = merged[[f'PC1_{spec_name}']]
            result = self._regression(y, X)
            result['model_name'] = f"dollar_log_return ~ PC1_{spec_name}"
            rows.append(result)

        self.regression_results = pd.DataFrame(rows)

    def export_latex(self, df: pd.DataFrame, filename: str, caption: str, label: str) -> None:
        """
        Export a DataFrame to a LaTeX-ready text file.

        This implementation avoids the optional Jinja2 dependency by rendering a
        simple tabular environment directly.
        """
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tables_dir / filename

        column_format = 'l' * len(df.columns)
        lines = [f"% {caption}", f"% label: {label}", "\\begin{tabular}{%s}" % column_format, "\\hline"]
        header = ' & '.join(df.columns.astype(str)) + ' \\\\'
        lines.append(header)
        lines.append('\\hline')

        for _, row in df.iterrows():
            formatted = []
            for value in row:
                if pd.isna(value):
                    formatted.append('')
                elif isinstance(value, (int, float, np.floating, np.integer)):
                    formatted.append(f"{value:.4f}")
                else:
                    formatted.append(str(value))
            lines.append(' & '.join(formatted) + ' \\\\')

        lines.append('\\hline')
        lines.append('\\end{tabular}')

        file_path.write_text('\n'.join(lines) + '\n')

    def _qualitative_interpretation(self, spec_name: str) -> str:
        """
        Build a short interpretation for a PCA specification.
        """
        explained = self.pca_results[spec_name]['explained_variance']
        pc1 = float(explained.loc[0, 'explained_variance_ratio'])
        pc2 = float(explained.loc[1, 'explained_variance_ratio'])
        cumulative = float(explained.loc[1, 'cumulative_explained_variance'])
        if pc1 >= 0.6 and cumulative >= 0.8:
            return 'PC1 appears to capture a common financial uncertainty/stress factor'
        if pc1 >= 0.4 and cumulative >= 0.75:
            return 'PC1 appears to capture a broad uncertainty/stress factor'
        return 'PC1 captures some common variation but the factor is mixed'

    def build_summary(self) -> None:
        """
        Build the compact PCA summary table.
        """
        rows = []
        for spec_name in ['A', 'B', 'C']:
            variables = self.pca_results[spec_name]['variables']
            explained = self.pca_results[spec_name]['explained_variance']
            pc1 = float(explained.loc[0, 'explained_variance_ratio'])
            pc2 = float(explained.loc[1, 'explained_variance_ratio'])
            cumulative = float(explained.loc[1, 'cumulative_explained_variance'])
            interpretation = self._qualitative_interpretation(spec_name)
            if spec_name == 'B':
                loadings = self.pca_results[spec_name]['loadings']
                stlfsi4_loading = abs(loadings.loc['STLFSI4_diff', 'PC1'])
                interpretation += ' STLFSI4 loads ' + ('strongly' if stlfsi4_loading >= 0.5 else 'weakly') + ' on PC1.'
            if spec_name == 'C':
                loadings = self.pca_results[spec_name]['loadings']
                gepu_loading = abs(loadings.loc['gepu_diff', 'PC1'])
                interpretation += ' GEPU loads ' + ('strongly' if gepu_loading >= 0.5 else 'weakly') + ' on PC1.'
            rows.append({
                'specification': spec_name,
                'variables_included': ', '.join(variables),
                'variance_explained_PC1': pc1,
                'variance_explained_PC2': pc2,
                'cumulative_variance_PC1_PC2': cumulative,
                'interpretation': interpretation
            })
        self.summary = pd.DataFrame(rows)

    def run_full_pipeline(self) -> None:
        """
        Run the full PCA pipeline, including PCA specs, regression diagnostics, and exports.
        """
        self.load_data()
        self.run_pca('A', ['vix_diff', 'vxy_g7_diff', 'vxy_em_diff'])
        self.run_pca('B', ['vix_diff', 'vxy_g7_diff', 'vxy_em_diff', 'STLFSI4_diff'])
        self.run_pca('C', ['vix_diff', 'vxy_g7_diff', 'vxy_em_diff', 'STLFSI4_diff', 'gepu_diff'])
        self.export_pca_outputs('A')
        self.export_pca_outputs('B')
        self.export_pca_outputs('C')
        self.build_summary()
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(self.tables_dir / 'pca_summary.csv', index=False)
        self.export_latex(self.summary, 'pca_summary_latex.txt', 'PCA Summary', 'tab:pca_summary')
        self.run_pca_regressions()
        self.regression_results.to_csv(self.tables_dir / 'pca_regressions.csv', index=False)
        self.export_latex(self.regression_results, 'pca_regressions_latex.txt', 'PCA Regression Results', 'tab:pca_reg')
