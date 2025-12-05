"""Data preprocessing Module for Insurance Analytics"""

import os
import numpy as np
from typing import List, Optional, Dict, Tuple
import pandas as pd


class DataPreprocessor:
    """
    Enhanced class-based preprocessor for insurance datasets with domain-specific
    validations, outlier detection, and feature engineering.

    Example:
        dp = DataPreprocessor(
            raw_path="D:/Python/Week-3/Raw_Data/MachineLearningRating_v3.txt",
            out_path="D:/Python/Week-3/Raw_Data/processed_MachineLearningRating_v3.txt",
            chunksize=100_000,
            delimiter="|"
        )
        dp.process(save_format="csv", create_features=True, run_quality_checks=True)
    """

    def __init__(
        self,
        raw_path: str,
        out_path: str,
        chunksize: int = 100_000,
        delimiter: str = "|",
        log_transform: bool = True
    ):
        self.raw_path = raw_path
        self.out_path = out_path
        self.chunksize = chunksize
        self.delimiter = delimiter
        self.log_transform = log_transform
        self.quality_issues = []
        self.transformation_log = []

    # -----------------------
    # Low-level helpers
    # -----------------------
    @staticmethod
    def _clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Trim whitespace, collapse repeated spaces and normalize strings (preserve NaN)."""
        obj_cols = df.select_dtypes(include=["object"]).columns

        for col in obj_cols:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
        return df

    @staticmethod
    def _convert_boolean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Map common truthy/falsy strings to booleans, leave NaN as-is."""
        mapping = {
            "yes": True, "y": True, "true": True, "1": True,
            "no": False, "n": False, "false": False, "0": False
        }

        for col in cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype("string")
                    .str.lower()
                    .map(mapping)
                )
        return df

    @staticmethod
    def _convert_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _convert_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col], errors="coerce", infer_datetime_format=True
                )
        return df

    @staticmethod
    def _convert_to_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    # -----------------------
    # Insurance-specific helpers
    # -----------------------
    def _validate_insurance_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct insurance business rules."""
        original_shape = df.shape

        # Premium should be positive
        if "TotalPremium" in df.columns:
            negative_premiums = (df["TotalPremium"] < 0).sum()
            if negative_premiums > 0:
                self.quality_issues.append(
                    f"Fixed {negative_premiums} negative premiums")
                df["TotalPremium"] = df["TotalPremium"].abs()

        # Claims should be non-negative
        if "TotalClaims" in df.columns:
            negative_claims = (df["TotalClaims"] < 0).sum()
            if negative_claims > 0:
                self.quality_issues.append(
                    f"Fixed {negative_claims} negative claims")
                df["TotalClaims"] = df["TotalClaims"].clip(lower=0)

        # Registration year shouldn't be in the future
        if "RegistrationYear" in df.columns:
            current_year = pd.Timestamp.now().year
            future_years = (df["RegistrationYear"] > current_year).sum()
            if future_years > 0:
                self.quality_issues.append(
                    f"Fixed {future_years} future registration years")
                df["RegistrationYear"] = df["RegistrationYear"].clip(
                    upper=current_year)

        # Vehicle age should be reasonable (0-50 years)
        if "RegistrationYear" in df.columns and "TransactionMonth" in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df["TransactionMonth"]):
                df["VehicleAge"] = (
                    df["TransactionMonth"].dt.year - df["RegistrationYear"]).clip(lower=0, upper=50)

        self.transformation_log.append(f"Validated {original_shape[0]} rows")
        return df

    def _handle_missing_insurance_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insurance-specific missing value imputation."""

        # For claims analysis, missing sum insured could be imputed with median by cover type
        if "SumInsured" in df.columns and "CoverType" in df.columns:
            missing_sum = df["SumInsured"].isna().sum()
            if missing_sum > 0:
                medians = df.groupby("CoverType")[
                    "SumInsured"].transform("median")
                df["SumInsured"] = df["SumInsured"].fillna(medians)
                self.transformation_log.append(
                    f"Imputed {missing_sum} missing SumInsured values")

        # For vehicle attributes, use median by vehicle type if available
        vehicle_cols = ["Cubiccapacity", "Kilowatts",
                        "NumberOfDoors", "Cylinders"]
        for col in vehicle_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    df[col] = df[col].fillna(df[col].median())
                    self.transformation_log.append(
                        f"Imputed {missing} missing {col} values")

        # For categorical columns, add 'Unknown' category
        cat_cols = ["Gender", "MaritalStatus", "Country", "Province"]
        for col in cat_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    if pd.api.types.is_categorical_dtype(df[col]):
                        df[col] = df[col].cat.add_categories(["Unknown"])
                    df[col] = df[col].fillna("Unknown")

        return df

    def _detect_insurance_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers specific to insurance data."""

        # Premium outliers (winsorize at 99.5th percentile)
        if "TotalPremium" in df.columns:
            upper_limit = df["TotalPremium"].quantile(0.995)
            extreme_premiums = (df["TotalPremium"] > upper_limit).sum()
            if extreme_premiums > 0:
                df["TotalPremium"] = df["TotalPremium"].clip(upper=upper_limit)
                self.quality_issues.append(
                    f"Winsorized {extreme_premiums} extreme premium values")

        # Claim amount outliers (winsorize at 99th percentile)
        if "TotalClaims" in df.columns:
            upper_limit = df["TotalClaims"].quantile(0.99)
            extreme_claims = (df["TotalClaims"] > upper_limit).sum()
            if extreme_claims > 0:
                df["TotalClaims"] = df["TotalClaims"].clip(upper=upper_limit)
                self.quality_issues.append(
                    f"Winsorized {extreme_claims} extreme claim values")

        # Sum insured should be reasonable (upper bound)
        if "SumInsured" in df.columns:
            upper_limit = df["SumInsured"].quantile(0.999)
            extreme_sums = (df["SumInsured"] > upper_limit).sum()
            if extreme_sums > 0:
                df["SumInsured"] = df["SumInsured"].clip(upper=upper_limit)

        return df

    def _create_insurance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features useful for insurance risk modeling."""

        # 1. Loss Ratio (key insurance metric)
        if all(col in df.columns for col in ["TotalClaims", "TotalPremium"]):
            df["LossRatio"] = df["TotalClaims"] / \
                (df["TotalPremium"].replace(0, np.nan) + 1e-10)
            df["LossRatio"] = df["LossRatio"].clip(
                lower=0, upper=10)  # Cap at 1000%

        # 2. Premium per unit sum insured (Rate)
        if all(col in df.columns for col in ["CalculatedPremiumPerTerm", "SumInsured"]):
            df["PremiumRate"] = df["CalculatedPremiumPerTerm"] / \
                (df["SumInsured"] + 1e-10)

        # 3. Boolean flag for claims presence
        if "TotalClaims" in df.columns:
            df["HasClaim"] = (df["TotalClaims"] > 0).astype("int8")

        # 4. Claim frequency indicator
        if "TotalClaims" in df.columns and "TotalPremium" in df.columns:
            df["ClaimIndicator"] = np.where(df["TotalClaims"] > 0, 1, 0)

        # 5. Log transformations for skewed monetary variables
        if self.log_transform:
            monetary_cols = ["TotalPremium", "TotalClaims",
                             "SumInsured", "CalculatedPremiumPerTerm"]
            for col in monetary_cols:
                if col in df.columns:
                    # Add small constant to avoid log(0)
                    df[f"Log_{col}"] = np.log1p(df[col].clip(lower=0))

        # 6. Vehicle age if not already calculated
        if "VehicleAge" not in df.columns and "RegistrationYear" in df.columns:
            if "TransactionMonth" in df.columns and pd.api.types.is_datetime64_any_dtype(df["TransactionMonth"]):
                df["VehicleAge"] = (
                    df["TransactionMonth"].dt.year - df["RegistrationYear"]).clip(lower=0)

        return df

    def _run_data_quality_checks(self, df: pd.DataFrame):
        """Run insurance-specific data quality checks."""

        print("\n" + "="*60)
        print("DATA QUALITY CHECKS")
        print("="*60)

        checks = []

        # Check 1: Policy dates consistency
        if "TransactionMonth" in df.columns:
            future_dates = (df["TransactionMonth"] > pd.Timestamp.now()).sum()
            if future_dates > 0:
                checks.append(
                    f"⚠️ {future_dates} future transaction dates found")

        # Check 2: Negative values in monetary columns
        monetary_cols = ["TotalPremium", "TotalClaims", "SumInsured"]
        for col in monetary_cols:
            if col in df.columns:
                negative = (df[col] < 0).sum()
                if negative > 0:
                    checks.append(f"⚠️ {negative} negative values in {col}")

        # Check 3: Claim > Sum Insured (possible data error)
        if all(col in df.columns for col in ["TotalClaims", "SumInsured"]):
            claim_exceed = (df["TotalClaims"] > df["SumInsured"]).sum()
            if claim_exceed > 0:
                checks.append(f"⚠️ {claim_exceed} claims exceed sum insured")

        # Check 4: Missing critical fields
        critical_fields = ["PolicyID", "TransactionMonth", "TotalPremium"]
        for field in critical_fields:
            if field in df.columns:
                missing_pct = df[field].isna().mean() * 100
                if missing_pct > 5:
                    checks.append(
                        f"⚠️ {missing_pct:.1f}% missing values in {field}")

        # Check 5: Duplicate policy IDs (if applicable)
        if "PolicyID" in df.columns:
            duplicates = df.duplicated(subset=["PolicyID"], keep=False).sum()
            if duplicates > 0:
                checks.append(f"⚠️ {duplicates} duplicate PolicyID entries")

        if checks:
            for check in checks:
                print(check)
        else:
            print("✅ All data quality checks passed")

        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)

        if "TotalPremium" in df.columns:
            print(f"Total Premium: ${df['TotalPremium'].sum():,.2f}")
            print(f"Average Premium: ${df['TotalPremium'].mean():,.2f}")

        if "TotalClaims" in df.columns:
            print(f"Total Claims: ${df['TotalClaims'].sum():,.2f}")
            print(f"Claim Frequency: {(df['TotalClaims'] > 0).mean():.2%}")

        if "LossRatio" in df.columns:
            print(f"Average Loss Ratio: {df['LossRatio'].mean():.2%}")

    # -----------------------
    # Chunk preprocessing
    # -----------------------
    def _preprocess_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the column-aware preprocessing to a single DataFrame chunk."""

        original_size = len(df)

        # 1) Policy fields
        for col in ["UnderwrittenCoverID", "PolicyID"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = self._convert_dates(df, ["TransactionMonth"])

        # 2) Client fields
        client_cat_cols = [
            "Citizenship", "LegalType", "Title", "Language",
            "Bank", "AccountType", "MaritalStatus", "Gender"
        ]
        df = self._convert_to_category(df, client_cat_cols)
        df = self._convert_boolean(df, ["IsVATRegistered"])

        # 3) Location
        df = self._convert_to_category(
            df, ["Country", "Province", "MainCrestaZone", "SubCrestaZone"]
        )
        if "PostalCode" in df.columns:
            df["PostalCode"] = df["PostalCode"].astype(str)

        # 4) Vehicle
        vehicle_numeric_cols = [
            "RegistrationYear", "Cylinders", "Cubiccapacity", "Kilowatts",
            "NumberOfDoors", "CustomValueEstimate", "CapitalOutstanding",
            "NumberOfVehiclesInFleet"
        ]
        df = self._convert_numeric(df, vehicle_numeric_cols)

        df = self._convert_boolean(
            df, ["NewVehicle", "WrittenOff",
                 "Rebuilt", "Converted", "CrossBorder"]
        )

        df = self._convert_dates(df, ["VehicleIntroDate"])

        # 5) Plan/Product
        df = self._convert_numeric(
            df, ["SumInsured", "CalculatedPremiumPerTerm", "ExcessSelected"]
        )

        df = self._convert_to_category(
            df,
            [
                "CoverCategory", "CoverType", "CoverGroup",
                "Section", "Product", "StatutoryClass", "StatutoryRiskType"
            ]
        )

        # 6) Premium & Claims
        df = self._convert_numeric(df, ["TotalPremium", "TotalClaims"])

        # 7) Insurance-specific validations and outlier handling
        df = self._validate_insurance_logic(df)
        df = self._detect_insurance_outliers(df)

        # 8) Handle missing values
        df = self._handle_missing_insurance_specific(df)

        # 9) Global cleaning
        df = self._clean_string_columns(df)

        # 10) Remove duplicates within chunk
        df.drop_duplicates(inplace=True)

        self.transformation_log.append(
            f"Chunk processed: {original_size} → {len(df)} rows")

        return df

    # -----------------------
    # I/O helpers
    # -----------------------
    def _iter_read_chunks(self):
        """Yield chunks safely from the raw text file."""
        return pd.read_csv(
            self.raw_path,
            sep=self.delimiter,
            chunksize=self.chunksize,
            low_memory=False,
            dtype=str,
            encoding="utf-8",
            encoding_errors="replace",
        )

    @staticmethod
    def _ensure_dir_for_path(path: str):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _save_dataframe(self, df: pd.DataFrame, path: str, save_fmt: str = "csv"):
        """Save df to path. save_fmt: 'csv' or 'parquet'."""
        self._ensure_dir_for_path(path)

        if save_fmt == "parquet":
            df.to_parquet(path, index=False, compression='snappy')
        else:
            df.to_csv(path, sep=self.delimiter, index=False)

    # -----------------------
    # Public entrypoint
    # -----------------------
    def process(self,
                save_format: str = "csv",
                sample_nrows: Optional[int] = None,
                create_features: bool = True,
                run_quality_checks: bool = True) -> pd.DataFrame:
        """
        Run enhanced preprocessing pipeline with insurance-specific steps.

        Args:
            save_format: 'csv' or 'parquet'
            sample_nrows: limit rows for quick testing
            create_features: whether to create derived insurance features
            run_quality_checks: whether to run data quality checks

        Returns:
            Final cleaned DataFrame
        """
        assert save_format in ("csv", "parquet")

        chunks = []
        processed_rows = 0

        print(f"🔄 Loading raw data from: {self.raw_path}")
        print(f"📊 Chunk size: {self.chunksize:,}")
        print(f"🎯 Target format: {save_format.upper()}")

        for i, chunk in enumerate(self._iter_read_chunks()):
            print(f"👉 Processing chunk {i+1} (size: {len(chunk):,})")

            cleaned = self._preprocess_chunk(chunk)
            chunks.append(cleaned)
            processed_rows += len(cleaned)

            if sample_nrows and processed_rows >= sample_nrows:
                print(
                    f"⏱ Sample limit reached ({sample_nrows} rows) — stopping early.")
                break

        print("🧹 Combining all cleaned chunks...")
        df_final = pd.concat(
            chunks, ignore_index=True) if chunks else pd.DataFrame()

        # Remove duplicates across chunks
        initial_rows = len(df_final)
        df_final.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df_final)
        if duplicates_removed > 0:
            self.transformation_log.append(
                f"Removed {duplicates_removed} duplicate rows across chunks")

        # Create insurance features on the final combined dataset
        if create_features:
            print("🔧 Creating insurance-specific features...")
            df_final = self._create_insurance_features(df_final)

        # Run data quality checks
        if run_quality_checks:
            self._run_data_quality_checks(df_final)

        # Print transformation log
        if self.transformation_log:
            print("\n" + "="*60)
            print("TRANSFORMATION LOG")
            print("="*60)
            for log_entry in self.transformation_log:
                print(f"• {log_entry}")

        # Print quality issues
        if self.quality_issues:
            print("\n" + "="*60)
            print("QUALITY ISSUES RESOLVED")
            print("="*60)
            for issue in self.quality_issues:
                print(f"• {issue}")

        # Auto-fix extension
        out_path = self.out_path
        if save_format == "parquet":
            if not out_path.lower().endswith(".parquet"):
                out_path = os.path.splitext(out_path)[0] + ".parquet"
        else:
            if not (out_path.lower().endswith(".txt") or out_path.lower().endswith(".csv")):
                out_path = os.path.splitext(out_path)[0] + ".txt"

        print(f"\n💾 Saving cleaned file to: {out_path}")
        self._save_dataframe(df_final, out_path, save_fmt=save_format)

        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Output file: {out_path}")
        print(f"📊 Final dataset shape: {df_final.shape}")
        print(
            f"📈 Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Display sample of new features
        if create_features:
            new_features = [col for col in df_final.columns if col.startswith(
                ('Log_', 'LossRatio', 'PremiumRate', 'HasClaim'))]
            if new_features:
                print(f"✨ New features created: {', '.join(new_features)}")

        return df_final


# -----------------------
# CLI runner
# -----------------------
if __name__ == "__main__":
    RAW_FILE_PATH = r"D:\Python\Week-3\Raw_Data\MachineLearningRating_v3.txt"
    OUTPUT_FILE_PATH = r"D:\Python\Week-3\Insurance-Analytics-Week-3-\data\processed\processed_MachineLearningRating_v3.csv"

    pre = DataPreprocessor(
        raw_path=RAW_FILE_PATH,
        out_path=OUTPUT_FILE_PATH,
        chunksize=100_000,
        delimiter="|",
        log_transform=True
    )

    # Process with enhanced features
    df_processed = pre.process(
        save_format="csv",
        create_features=True,
        run_quality_checks=True
    )

    # Optional: Display sample of processed data
    print("\n" + "="*60)
    print("SAMPLE OF PROCESSED DATA")
    print("="*60)
    print(df_processed.head())
