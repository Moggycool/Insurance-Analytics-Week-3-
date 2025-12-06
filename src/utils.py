"""
Utility functions for Insurance Analytics
Converted to class-based design
"""

import os
import pandas as pd
import numpy as np


class DataUtils:
    """Utility helper class for file handling and DataFrame summarization."""

    @staticmethod
    def ensure_dir(path: str):
        """Ensure the directory exists."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: str, fmt: str = "csv", delimiter: str = "|"):
        """
        Save a DataFrame to CSV or Parquet.
        """
        DataUtils.ensure_dir(path)
        fmt = fmt.lower()

        if fmt == "parquet":
            if not path.endswith(".parquet"):
                path = os.path.splitext(path)[0] + ".parquet"
            df.to_parquet(path, index=False, compression="snappy")

        else:
            # default to csv/txt
            if not path.endswith(".csv") and not path.endswith(".txt"):
                path = os.path.splitext(path)[0] + ".txt"
            df.to_csv(path, sep=delimiter, index=False)

    @staticmethod
    def memory_usage(df: pd.DataFrame, deep: bool = True) -> float:
        """Return memory usage of a DataFrame in MB."""
        return df.memory_usage(deep=deep).sum() / 1024 ** 2

    @staticmethod
    def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns summary table:
        - dtype
        - missing count & %
        - unique values
        - top frequency
        """
        summary = pd.DataFrame({
            "dtype": df.dtypes,
            "missing": df.isna().sum(),
            "unique": df.nunique(),
            "top_freq": df.apply(
                lambda x: x.value_counts().max() if x.nunique() > 0 else np.nan
            )
        })
        summary["missing_pct"] = summary["missing"] / len(df) * 100
        return summary
