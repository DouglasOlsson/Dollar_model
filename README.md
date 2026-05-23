# Dollar Model

This repository contains the Python code used for the empirical analysis in a bachelor thesis on short-run U.S. dollar appreciation and global financial uncertainty.

The project investigates whether changes in global financial uncertainty proxies are associated with monthly U.S. dollar appreciation over the period 2006–2025. The dependent variable is the monthly log return of the Nominal Broad U.S. Dollar Index (DTWEXBGS). The explanatory variables represent market-implied volatility, financial stress, and policy uncertainty.

## Project structure

```text
model_dollar/
├── data/
│   ├── raw/              # Original input data
│   ├── interim/          # Cleaned intermediate datasets
│   └── processed/        # Final datasets used for analysis
├── figures/              # Generated figures
├── output/
│   └── tables/           # Regression tables and statistical outputs
├── scripts/              # Scripts for running each analysis step
├── src/                  # Reusable processing modules
├── .gitignore
└── README.md
```

## Data

The analysis uses monthly data from several sources:

- **DTWEXBGS:** Nominal Broad U.S. Dollar Index
- **VIX:** equity-market implied volatility
- **VXY-G7:** G7 foreign-exchange implied volatility
- **VXY-EM:** emerging-market foreign-exchange implied volatility
- **STLFSI4:** St. Louis Fed Financial Stress Index
- **GEPU:** Global Economic Policy Uncertainty Index

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

## How to run the project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy statsmodels matplotlib scikit-learn openpyxl
```

Run the scripts in the following order:

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

## Main output files

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

## Reproducibility notes

The code is structured so that the main empirical results can be reproduced from the raw data files by running the scripts in sequence. Intermediate and processed datasets are included to make the workflow transparent.

Some data sources may be subject to access restrictions or licensing terms. In particular, foreign-exchange implied volatility series such as VXY-G7 and VXY-EM may require access through financial data providers.

## Author

Douglas Olsson
