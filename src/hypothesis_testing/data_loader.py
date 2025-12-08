import pandas as pd
import os


class DataLoader:
    def __init__(self, file_path: str = None):
        # Resolve project root robustly
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        default_path = os.path.join(
            project_root,
            "data",
            "processed",
            "processed_MachineLearningRating_v3.csv"
        )

        # If caller passes a bad relative path, prefer the safe default
        if file_path and os.path.isabs(file_path):
            self.file_path = file_path
        else:
            self.file_path = default_path

        self.df = None

    def load(self):
        """
        Load pipe-separated dataset safely.
        Fixes the issue where all columns were read as one.
        """
        os.makedirs("reports", exist_ok=True)

        self.df = pd.read_csv(
            self.file_path,
            sep="|",
            engine="python",
            on_bad_lines="skip"  # robust against malformed rows
        )

        return self.df

    def create_metrics(self):
        """
        Create KPI metrics:
        - Claim Frequency
        - Claim Severity
        - Margin
        """
        df = self.df.copy()

        # Safely convert numeric columns
        for col in ["TotalPremium", "TotalClaims"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0.0

        # Claim Frequency
        if "HasClaim" not in df.columns:
            df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

        # Claim Severity
        df["ClaimSeverity"] = df["TotalClaims"].where(
            df["HasClaim"] == 1, 0
        )

        # Margin
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

        self.df = df
        return df
