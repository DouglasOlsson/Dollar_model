# Dollar Model

This repository contains the Python code used for the empirical analysis in a bachelor thesis on short-run U.S. dollar appreciation and global financial uncertainty.

The project investigates whether changes in global financial uncertainty proxies are associated with monthly U.S. dollar appreciation over the period 2006–2025. The dependent variable is the monthly log return of the Nominal Broad U.S. Dollar Index (DTWEXBGS). The explanatory variables represent market-implied volatility, financial stress, and policy uncertainty.

## Project Structure

```text
model_dollar/
├── data/
│   ├── raw/              # Original input data, not included in the public repository
│   ├── interim/          # Cleaned intermediate datasets, not included in the public repository
│   └── processed/        # Final datasets used for analysis, not included in the public repository
├── figures/              # Generated figures
├── output/
│   └── tables/           # Regression tables and statistical outputs
├── scripts/              # Scripts for running each analysis step
├── src/                  # Reusable processing modules
├── .gitignore
└── README.md
```

## Data Availability

The repository does **not** include raw, interim, or processed data files.

This is intentional. Some of the data used in the thesis may be subject to access restrictions or licensing terms. In particular, the foreign-exchange implied volatility series **VXY-G7** and **VXY-EM** were obtained through LSEG Workspace and are therefore not redistributed in this repository.

Users who want to reproduce the full analysis must obtain the required data separately from the relevant providers and place the files in the appropriate local data folders.

The analysis uses monthly data from the following sources:

| Variable | Description | Source |
|---|---|---|
| DTWEXBGS | Nominal Broad U.S. Dollar Index | FRED |
| VIX | Equity-market implied volatility | Yahoo Finance / CBOE |
| VXY-G7 | G7 foreign-exchange implied volatility | LSEG Workspace |
| VXY-EM | Emerging-market foreign-exchange implied volatility | LSEG Workspace |
| STLFSI4 | St. Louis Fed Financial Stress Index | FRED |
| GEPU | Global Economic Policy Uncertainty Index | PolicyUncertainty.com |

The dependent variable is calculated as the monthly log return of the dollar index. The explanatory variables are transformed into first differences to align the empirical specification with short-run changes in financial uncertainty.

## Methodology

The empirical analysis includes:

- data cleaning and monthly alignment
- stationarity testing using Augmented Dickey-Fuller tests
- bivariate OLS regressions
- multivariate OLS regressions
- Newey-West HAC standard errors
- regime and crisis-period analysis
- structural break tests
- principal component analysis as a complementary analysis

## How to Run the Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy statsmodels matplotlib scikit-learn openpyxl
```

Place the required raw data files locally in:

```text
data/raw/
```

Then run the scripts in the following order:

```bash
python scripts/build_master_monthly.py
python scripts/build_master_monthly_stationary.py
python scripts/run_adf_tests.py
python scripts/run_bivariate_regressions.py
python scripts/run_sq2_multivariate_regressions.py
python scripts/run_sq3_regime_analysis.py
python scripts/run_structural_break_tests.py
python scripts/run_pca_analysis.py
python scripts/generate_stationary_descriptives.py
python scripts/generate_method_figures.py
```

Generated results are saved in:

```text
output/tables/
figures/
```

## Main Output Files

Important output files include:

```text
output/tables/adf_results_stationarity.csv
output/tables/bivariate_regressions_stationary.csv
output/tables/bivariate_regression_ranking.csv
output/tables/sq2_multivariate_regressions.csv
output/tables/sq3_interaction_models.csv
output/tables/sq3_subperiod_regressions.csv
output/tables/stationary_descriptive_stats.csv
output/tables/stationary_correlation_matrix.csv
output/tables/pca_summary.csv
```

LaTeX versions of selected tables are also generated as `.txt` files.

## Reproducibility Notes

The code is structured so that the main empirical workflow can be reproduced from the required input data by running the scripts in sequence.

Because access-restricted data are not included, the public repository should be understood as a reproducible code repository rather than a complete open-data archive. Users with access to the same data sources can reproduce the full analysis by placing the required files in the expected local folders.

The repository excludes:

```text
data/raw/
data/interim/
data/processed/
*.xlsx
*.csv
```

These exclusions are defined in `.gitignore` to avoid redistributing raw or processed data files, including licensed financial data.

## Author

Douglas Olsson
