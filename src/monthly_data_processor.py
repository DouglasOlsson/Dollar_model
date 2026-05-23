import pandas as pd
import pathlib
from typing import Dict


class MonthlyDataProcessor:
    """
    A class for preprocessing monthly financial time series data.

    This class handles loading, cleaning, standardizing, and merging multiple
    monthly financial datasets from CSV and Excel files. It standardizes dates
    to the first day of each month, detects appropriate value columns, and
    produces cleaned individual series files plus a merged master dataset.
    """

    def __init__(self, config: Dict[str, Dict[str, str]]):
        """
        Initialize the processor with a configuration dictionary.

        Args:
            config: Dictionary mapping relative file paths to series info.
                   Keys are file paths, values are dicts with 'name' and 'end_date'.
        """
        self.config = config
        self.cleaned_data: Dict[str, pd.DataFrame] = {}

    def detect_date_column(self, df: pd.DataFrame) -> str:
        """
        Detect the date column in the DataFrame.

        Checks for common date column names or infers from data types.
        If 'Year' and 'Month' columns exist, returns 'Date' (to be created later).

        Args:
            df: Input DataFrame.

        Returns:
            Name of the date column ('Date').

        Raises:
            ValueError: If no suitable date column is found.
        """
        # Special case: if Year and Month columns exist, will create Date
        if 'Year' in df.columns and 'Month' in df.columns:
            return 'Date'

        possible_names = ['Date', 'date', 'DATE', 'Time', 'time', 'Exchange Date']
        for name in possible_names:
            if name in df.columns:
                return name

        # Check for datetime-like columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
            # Check if object column can be parsed as dates
            if df[col].dtype == 'object':
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > len(df) * 0.8:
                        return col
                except:
                    continue

        raise ValueError("No suitable date column found in the file.")

    def detect_value_column(self, df: pd.DataFrame, series_name: str) -> str:
        """
        Detect the primary value column for the series.

        For price-market style data, prefers 'Adj Close', then 'Close'.
        For macro/index series, looks for the series name or a 'Value' column.

        Args:
            df: Input DataFrame.
            series_name: Name of the series (e.g., 'dollar').

        Returns:
            Name of the value column.

        Raises:
            ValueError: If no suitable value column is found.
        """
        # Priority for OHLC-style data
        if 'Adj Close' in df.columns:
            return 'Adj Close'
        if 'Close' in df.columns:
            return 'Close'

        # For macro series, check series name or common value names
        if series_name in df.columns:
            return series_name
        # Special cases for known series
        if series_name == 'gepu' and 'GEPU_current' in df.columns:
            return 'GEPU_current'
        if 'Value' in df.columns:
            return 'Value'
        if 'value' in df.columns:
            return 'value'

        # Fallback: find the first numeric column excluding Date
        date_col = self.detect_date_column(df)
        for col in df.columns:
            if col != date_col and col != 'Date' and pd.api.types.is_numeric_dtype(df[col]):
                return col

        raise ValueError(f"No suitable value column found for series '{series_name}'.")

    def detect_header_row(self, df_no_header: pd.DataFrame) -> int:
        """
        Detect the header row in an Excel DataFrame read without header.

        Looks for rows where the first column contains date-like or header-like strings.

        Args:
            df_no_header: DataFrame read with header=None.

        Returns:
            Row index of the header.
        """
        possible_headers = ['Exchange Date', 'Date', 'Year', 'Month', 'Time']
        for i in range(min(50, len(df_no_header))):  # Check first 50 rows
            first_col = str(df_no_header.iloc[i, 0]).strip()
            if first_col in possible_headers:
                return i
        # Fallback to row 0
        return 0

    def load_file(self, file_path: pathlib.Path) -> pd.DataFrame:
        """
        Load a CSV or Excel file into a DataFrame.

        For Excel files, detects the correct header row.

        Args:
            file_path: Path to the file.

        Returns:
            Loaded DataFrame.

        Raises:
            ValueError: If file type is unsupported.
            FileNotFoundError: If file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.xlsx':
            # First read without header to detect header row
            df_temp = pd.read_excel(file_path, sheet_name=0, header=None)
            header_row = self.detect_header_row(df_temp)
            df = pd.read_excel(file_path, sheet_name=0, header=header_row)
            print(f"Excel file {file_path.name}: detected header at row {header_row}")
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        return df

    def standardize_monthly_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize dates to the first day of each month.

        Args:
            df: DataFrame with a 'Date' column.

        Returns:
            DataFrame with standardized 'Date' column.
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        return df

    def clean_series(self, file_path: pathlib.Path, series_name: str, end_date: str) -> pd.DataFrame:
        """
        Load, clean, and standardize a single series.

        - Loads the file
        - Detects and renames date and value columns
        - Standardizes dates to month start
        - Filters to date range 2006-01-01 to 2026-01-01
        - Keeps only Date and value columns
        - Drops duplicates (keeps last occurrence, assuming most recent)
        - Sorts by Date

        Args:
            file_path: Path to the input file.
            series_name: Target name for the series.
            end_date: End date string for filtering (inclusive).

        Returns:
            Cleaned DataFrame.
        """
        df = self.load_file(file_path)

        print(f"Loaded {file_path.name}: columns {df.columns.tolist()[:5]}...")  # Debug

        # Detect and standardize column names
        date_col = self.detect_date_column(df)
        print(f"Detected date column: {date_col}")  # Debug
        if date_col == 'Date' and 'Year' in df.columns and 'Month' in df.columns:
            df = df.dropna(subset=['Year', 'Month'])
            df['Date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + df['Month'].astype(int).astype(str) + '-01')
        else:
            df = df.rename(columns={date_col: 'Date'})

        value_col = self.detect_value_column(df, series_name)
        print(f"Detected value column: {value_col} for series {series_name}")  # Debug
        df = df.rename(columns={value_col: series_name})

        # Standardize dates
        df = self.standardize_monthly_date(df)

        # Keep only relevant columns
        df = df[['Date', series_name]]

        # Filter date range
        start_date = pd.Timestamp('2006-01-01')
        end_date_ts = pd.Timestamp(end_date)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date_ts)]

        # Drop duplicates: keep last (most recent) observation per month
        df = df.drop_duplicates(subset='Date', keep='last')

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        print(f"After cleaning: {len(df)} rows")  # Debug

        self.cleaned_data[series_name] = df
        return df

    def save_clean_series(self, series_name: str, output_dir: pathlib.Path) -> None:
        """
        Save a cleaned series to CSV.

        Args:
            series_name: Name of the series.
            output_dir: Directory to save the file.
        """
        df = self.cleaned_data[series_name]
        output_path = output_dir / f"{series_name}_1mo_clean.csv"
        df.to_csv(output_path, index=False)

    def merge_series(self) -> pd.DataFrame:
        """
        Merge all cleaned series into a master dataset using outer join.

        Returns:
            Merged DataFrame.
        """
        if not self.cleaned_data:
            raise ValueError("No cleaned data available. Run clean_series first.")

        dfs = list(self.cleaned_data.values())
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on='Date', how='outer')

        # Sort by date
        merged = merged.sort_values('Date').reset_index(drop=True)
        return merged

    def print_summary(self) -> None:
        """
        Print a summary of the processing results.
        """
        print("\n--- Processing Summary ---")
        for series_name, df in self.cleaned_data.items():
            row_count = len(df)
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            missing_count = df[series_name].isna().sum()
            duplicate_check = df['Date'].duplicated().sum()
            print(f"{series_name}: {row_count} rows, {min_date} to {max_date}, "
                  f"{missing_count} missing values, {duplicate_check} duplicate dates")

        merged = self.merge_series()
        print(f"\nMerged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns, "
              f"dates {merged['Date'].min()} to {merged['Date'].max()}")

    def run(self, input_dir: pathlib.Path, interim_dir: pathlib.Path, processed_dir: pathlib.Path) -> None:
        """
        Run the full preprocessing pipeline.

        Args:
            input_dir: Directory containing input files.
            interim_dir: Directory to save individual cleaned files.
            processed_dir: Directory to save the merged master file.
        """
        # Process each series
        for file_name, series_info in self.config.items():
            series_name = series_info['name']
            end_date = series_info['end_date']
            file_path = input_dir / file_name
            print(f"Processing {series_name} from {file_path}...")
            self.clean_series(file_path, series_name, end_date)
            self.save_clean_series(series_name, interim_dir)

        # Merge and save master dataset
        print("Merging series...")
        merged = self.merge_series()
        # Filter master to end at 2025-12-01
        merged = merged[merged['Date'] <= pd.Timestamp('2025-12-01')]
        master_path = processed_dir / "master_monthly.csv"
        merged.to_csv(master_path, index=False)

        # Print summary
        self.print_summary()