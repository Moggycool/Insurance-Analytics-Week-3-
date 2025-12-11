"""
Module for preparing claim data for modeling.
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional, Any
import re
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClaimDataPreparer:
    """
    Prepares claim data for modeling with comprehensive preprocessing,
    feature engineering, and train-test splitting.

    NOTE:
    - This module DOES NOT perform target encoding. Categorical columns are
      preserved (as object/category). Apply cross-validated target encoding
      later in your modeling pipeline (Option B).
    """

    def __init__(self,
                 claim_data_path: str = None,
                 df_claims: pd.DataFrame = None,
                 target_col: str = 'TotalClaims',
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the data preparer.

        Args:
            claim_data_path: Path to extracted claim data CSV
            df_claims: DataFrame with claim data (alternative to file path)
            target_col: Name of the target variable column
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
        """
        if claim_data_path:
            self.claim_data_path = Path(claim_data_path)
            try:
                logger.info(
                    "📄 Loading CSV with robust parser (engine='python', skip bad lines)")
                self.df = pd.read_csv(
                    self.claim_data_path, engine='python', sep='|', on_bad_lines='skip')
            except Exception as e:
                logger.warning(
                    f"⚠ CSV load with sep='|' failed. Retrying with default parser. Error: {e}"
                )
            logger.info(
                f"Loaded dataset with {len(self.df):,} rows after skipping malformed rows.")
        elif df_claims is not None:
            self.df = df_claims.copy()
            self.claim_data_path = None
        else:
            raise ValueError(
                "Either claim_data_path or df_claims must be provided")

        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        # Initialize attributes
        self.df_prepared: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.label_encoders: Dict[str, Any] = {}

        # Configuration
        self.missing_threshold = 0.5  # Remove columns with >50% missing
        self.max_categories = 50  # Informational only

        logger.info(
            f"Initialized ClaimDataPreparer with {len(self.df)} records")
        logger.info(f"Target variable: {self.target_col}")
        logger.info(f"Test size: {test_size}")

    # -----------------------
    # Utilities
    # -----------------------
    def _clean_numeric_string(self, value):
        """Clean numeric strings by extracting numbers from text."""
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.number)):
            return float(value)

        if isinstance(value, str):
            v = value.strip()
            # Handle European decimals (comma)
            if ',' in v and '.' not in v:
                if v.count(',') == 1 and len(v.split(',')[-1]) <= 2:
                    v = v.replace(',', '.')
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', v)
            if numbers:
                try:
                    return float(numbers[0])
                except Exception:
                    return np.nan
        return np.nan

    # -----------------------
    # Main pipeline
    # -----------------------
    def prepare_data(self, save_path: str = None) -> pd.DataFrame:
        """
        Perform all data preparation steps.

        Args:
            save_path: Path to save prepared data (optional)

        Returns:
            Prepared DataFrame (df_prepared)
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("=" * 60)

        # Start with original data
        df = self.df.copy()

        # Step 0: Clean numeric-like columns first
        df_cleaned = self._clean_numeric_columns(df)

        # ============================================================
        # FIX TARGET: Remove invalid negative or null TotalClaims
        # ============================================================
        invalid_before = df_cleaned["TotalClaims"].isna(
        ).sum() + (df_cleaned["TotalClaims"] < 0).sum()
        print(
            f"🔎 Removing {invalid_before} invalid TotalClaims records (negative or NaN)")

        df_cleaned = df_cleaned[df_cleaned["TotalClaims"].notna() & (
            df_cleaned["TotalClaims"] >= 0)].copy()

        # ============================================================
        # SAFE CLAIM SEVERITY CATEGORY (5 bins)
        # Handles skew & duplicates
        # ============================================================
        print("🏷 Rebuilding ClaimSeverityCategory safely (Option B robust fix)")

        df_cleaned["ClaimSeverityCategory"] = pd.qcut(df_cleaned["TotalClaims"].rank(method="first"),
                                                      q=5,
                                                      labels=[
                                                          "VeryLow", "Low", "Medium", "High", "VeryHigh"],
                                                      duplicates="drop"
                                                      )
        # ============================================================
        # STRATIFICATION BIN — ROBUST 10-DECILE METHOD WITH FALLBACK
        # ============================================================
        print("📊 Creating stratification bin (no NaNs guaranteed)")
        df_cleaned["StratifyBin"] = pd.qcut(df_cleaned["TotalClaims"].rank(method="first"),
                                            q=10,
                                            labels=False,
                                            duplicates="drop")

        # Fallback if qcut still produces NaN
        if df_cleaned["StratifyBin"].isna().any():
            print("⚠️ qcut produced NaN — applying fallback using pd.cut")
            df_cleaned["StratifyBin"] = pd.cut(df_cleaned["TotalClaims"],
                                               bins=10,
                                               labels=False)

        # Step 1: Initial data inspection
        self._inspect_data(df_cleaned)

        # Step 2: Handle missing values and basic imputations
        df_missing_handled = self._handle_missing_data(df_cleaned)

        # Step 3: Feature engineering (keeps categories intact)
        df_engineered = self._feature_engineering(df_missing_handled)

        # Step 4: Sanitize categorical columns (no encoding)
        df_sanitized = self._encode_categorical_data(df_engineered)

        # Step 5: Create target variables (log, binary, categories). DOES NOT DROP ROWS.
        df_targets = self._create_target_variables(df_sanitized)

        # Step 6: Remove unnecessary columns (identifiers / single value). Do NOT drop high-cardinality categories.
        df_final = self._remove_unnecessary_columns(df_targets)

        # Set df_prepared BEFORE splitting and saving
        self.df_prepared = df_final.copy()

        # Update column lists
        self.categorical_columns = self.get_categorical_cols(self.df_prepared)
        self.numerical_columns = self.get_numerical_cols(self.df_prepared)

        # Step 7: Split into train/test (stratify using a binned regression target)
        self._train_test_split(self.df_prepared)

        # Step 8: Save if requested
        if save_path:
            self.save_prepared_data(save_path)

        logger.info("✅ Data preparation complete!")
        return self.df_prepared

    # -----------------------
    # Cleaning steps
    # -----------------------
    # def _clean_numeric_columns(self) -> pd.DataFrame:
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns that may contain text or mixed formats."""
        logger.info("\n🧹 STEP 0: CLEANING NUMERIC COLUMNS")
        df = self.df.copy()

        potential_numeric_cols = [
            'RegistrationYear', 'cubiccapacity', 'kilowatts',
            'CapitalOutstanding', 'SumInsured', 'TotalPremium',
            'TotalClaims', 'ExcessSelected', 'CalculatedPremiumPerTerm'
        ]

        for col in potential_numeric_cols:
            if col in df.columns:
                original_dtype = df[col].dtype
                original_non_nan = df[col].notna().sum()
                df[col] = df[col].apply(self._clean_numeric_string)
                converted_non_nan = df[col].notna().sum()
                logger.info(
                    f"  🔧 {col}: {original_dtype} | non-null {original_non_nan} -> {converted_non_nan}")

        return df

    def _inspect_data(self, df: pd.DataFrame) -> None:
        """Perform initial data inspection and basic sanity checks."""
        logger.info("\n📊 STEP 1: DATA INSPECTION")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in data")

        # Log basic target stats (safe to compute)
        try:
            logger.info(
                f"Target '{self.target_col}' stats — min: {df[self.target_col].min()}, max: {df[self.target_col].max()}, mean: {df[self.target_col].mean():.2f}, std: {df[self.target_col].std():.2f}")
        except Exception:
            logger.info("Could not compute target summary (non-numeric?)")

        # Missing value summary
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].sort_values(
            ascending=False)
        if not missing_cols.empty:
            logger.info("Missing values (top):")
            for col, cnt in missing_cols.head(20).items():
                logger.info(f"  - {col}: {cnt} missing")

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values:
         - Drop columns with > missing_threshold proportion missing
         - Impute numeric with median
         - Fill categorical with 'Unknown'
        """
        logger.info("\n🔧 STEP 2: HANDLING MISSING DATA")
        original_shape = df.shape

        missing_pct = df.isnull().mean()
        cols_to_drop = missing_pct[missing_pct >
                                   self.missing_threshold].index.tolist()
        if cols_to_drop:
            logger.info(
                f"Dropping columns with >{self.missing_threshold*100:.0f}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Identify columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

        # Do not remove target from num list here
        if self.target_col in num_cols:
            # keep it in num_cols but ensure we don't impute target blindly
            num_cols_no_target = [c for c in num_cols if c != self.target_col]
        else:
            num_cols_no_target = num_cols

        # Impute numeric columns with median
        for col in num_cols_no_target:
            missing = df[col].isna().sum()
            if missing:
                med = df[col].median()
                df[col] = df[col].fillna(med)
                logger.info(
                    f"  Imputed {missing} missing in {col} with median {med}")

        # Fill categorical columns with 'Unknown'
        for col in cat_cols:
            missing = df[col].isna().sum()
            if missing:
                if pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].cat.add_categories(['Unknown'])
                df[col] = df[col].fillna('Unknown')
                logger.info(
                    f"  Filled {missing} missing in {col} with 'Unknown'")

        return df

    # -----------------------
    # Feature engineering
    # -----------------------

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n⚙️ STEP 3: FEATURE ENGINEERING")

        df = df.copy()
        df = df[df["TotalClaims"] >= 0].copy()
        original_cols = df.columns.tolist()

        # TransactionMonth -> temporal features
        if 'TransactionMonth' in df.columns:
            try:
                df['TransactionMonth'] = pd.to_datetime(
                    df['TransactionMonth'], errors='coerce', infer_datetime_format=True)
                df['TransactionYear'] = df['TransactionMonth'].dt.year
                df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
                df['TransactionQuarter'] = df['TransactionMonth'].dt.quarter
                df['IsYearEnd'] = df['TransactionMonthNum'].isin(
                    [11, 12]).astype(int)
            except Exception as e:
                logger.warning(f"TransactionMonth parse warning: {e}")

        # Vehicle age
        if 'RegistrationYear' in df.columns:
            ref_year = df['TransactionYear'].max(
            ) if 'TransactionYear' in df.columns else pd.Timestamp.now().year
            df['VehicleAge'] = (
                ref_year - df['RegistrationYear']).clip(lower=0, upper=50)
            df['IsOldVehicle'] = (df['VehicleAge'] > 10).astype(int)
            df['IsNewVehicle'] = (df['VehicleAge'] < 3).astype(int)

        # Engine / power features
        if 'cubiccapacity' in df.columns and 'kilowatts' in df.columns:
            df['PowerToCapacityRatio'] = df['kilowatts'] / \
                (df['cubiccapacity'].replace(0, np.nan) + 1)
            df['HasHighPower'] = (
                df['PowerToCapacityRatio'] > df['PowerToCapacityRatio'].median()).astype(int)

        # Premium risk features
        if 'SumInsured' in df.columns and 'TotalPremium' in df.columns:
            df['PremiumToSumInsuredRatio'] = df['TotalPremium'] / \
                (df['SumInsured'].replace(0, np.nan) + 1)
            df['PremiumToSumInsuredRatio'] = df['PremiumToSumInsuredRatio'].replace(
                [np.inf, -np.inf], np.nan).fillna(df['PremiumToSumInsuredRatio'].median())
            df['IsHighRiskPremium'] = (
                df['PremiumToSumInsuredRatio'] > df['PremiumToSumInsuredRatio'].quantile(0.75)).astype(int)

        # ExcessAmount and HasHighExcess
        if 'ExcessSelected' in df.columns:
            if pd.api.types.is_numeric_dtype(df['ExcessSelected']):
                df['ExcessAmount'] = df['ExcessSelected']
            else:
                df['ExcessAmount'] = df['ExcessSelected'].astype(
                    str).str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
            df['ExcessAmount'] = df['ExcessAmount'].fillna(
                df['ExcessAmount'].median())
            df['HasHighExcess'] = (df['ExcessAmount'] >
                                   df['ExcessAmount'].median()).astype(int)

        # Security
        security_cols = [c for c in ['AlarmImmobiliser',
                                     'TrackingDevice'] if c in df.columns]
        if security_cols:
            df['HasSecurity'] = 0
            for c in security_cols:
                df['HasSecurity'] += df[c].notna().astype(int)
            df['HasSecurity'] = df['HasSecurity'].clip(upper=2)
            df['HasAdvancedSecurity'] = (df['HasSecurity'] == 2).astype(int)

        # Geographic risk (simple heuristic)
        if 'MainCrestaZone' in df.columns:
            df['IsHighRiskZone'] = df['MainCrestaZone'].astype(str).str.contains(
                r'(A|1|High)', case=False, na=False).astype(int)

        # Top make indicator
        if 'make' in df.columns:
            top_makes = df['make'].value_counts().nlargest(10).index.tolist()
            df['IsTopMake'] = df['make'].isin(top_makes).astype(int)

        # Coverage
        if 'CoverType' in df.columns:
            df['IsComprehensive'] = df['CoverType'].astype(str).str.contains(
                'Comprehensive', case=False, na=False).astype(int)

        # Interaction features
        if 'VehicleAge' in df.columns and 'SumInsured' in df.columns:
            df['AgeValueInteraction'] = df['VehicleAge'] * df['SumInsured']

        # Capital features
        if 'CapitalOutstanding' in df.columns:
            df['HasCapitalOutstanding'] = (
                df['CapitalOutstanding'] > 0).astype(int)
            df['CapitalToSumInsuredRatio'] = df['CapitalOutstanding'] / \
                (df['SumInsured'].replace(0, np.nan) + 1)
            df['CapitalToSumInsuredRatio'] = df['CapitalToSumInsuredRatio'].replace(
                [np.inf, -np.inf], np.nan).fillna(0)

        # Monetary log transforms (safe: log1p supports zeros)
        monetary_cols = [c for c in ['TotalPremium', 'TotalClaims',
                                     'SumInsured', 'CalculatedPremiumPerTerm'] if c in df.columns]
        for c in monetary_cols:
            df[f'Log_{c}'] = np.log1p(df[c].clip(lower=0))

        new_cols = [c for c in df.columns if c not in original_cols]
        logger.info(
            f"Created {len(new_cols)} new features: {new_cols[:10]}{'...' if len(new_cols) > 10 else ''}")

        return df

    # -----------------------
    # Categorical handling (no encoding)
    # -----------------------
    def _encode_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize categorical columns but DO NOT encode them here.
        Keep object/category columns intact for downstream CV target encoding.
        """
        logger.info("\n🔢 STEP 4: SANITIZING CATEGORICAL COLUMNS (no encoding)")
        df = df.copy()
        cat_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

        for col in cat_cols:
            # Trim whitespace & coerce to string, keep NaN replaced earlier
            df[col] = df[col].astype(
                str).str.strip().replace({'nan': 'Unknown'})
            # Optionally convert to 'category' dtype for memory efficiency
            try:
                df[col] = df[col].astype('category')
            except Exception:
                pass

        logger.info(f"Categorical columns preserved: {len(cat_cols)}")
        return df

    # -----------------------
    # Target creation (no row drops)
    # -----------------------
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target transformations without filtering rows (no leakage).
        - Log_TotalClaims: log1p(TotalClaims)
        - Sqrt_TotalClaims: sqrt(TotalClaims)
        - Std_TotalClaims: standardized (z-score)
        - HighClaim: binary based on median
        - ClaimSeverityCategory: terciles (Low/Medium/High)
        """
        logger.info("\n🎯 STEP 5: CREATING TARGET VARIABLES (no row removal)")
        df = df.copy()

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in data")

        # ensure numeric
        df[self.target_col] = pd.to_numeric(
            df[self.target_col], errors='coerce').fillna(0.0)

        # transformations (log1p handles 0)
        df['Log_TotalClaims'] = np.log1p(df[self.target_col])
        df['Sqrt_TotalClaims'] = np.sqrt(df[self.target_col].clip(lower=0))
        mean_val = df[self.target_col].mean()
        std_val = df[self.target_col].std()
        df['Std_TotalClaims'] = (
            df[self.target_col] - mean_val) / (std_val + 1e-10)

        # binary classification target
        median_claim = df[self.target_col].median()
        df['HighClaim'] = (df[self.target_col] > median_claim).astype(int)

        # claim severity categories (terciles) - uses quantiles on non-negative values
        try:
            q1, q2 = df[self.target_col].quantile([0.33, 0.66]).values
            df['ClaimSeverityCategory'] = pd.cut(
                df[self.target_col], bins=[-np.inf, q1, q2, np.inf], labels=['Low', 'Medium', 'High'])
        except Exception:
            df['ClaimSeverityCategory'] = pd.cut(
                df[self.target_col], bins=3, labels=['Low', 'Medium', 'High'])

        logger.info(
            "Target variables created: Log_TotalClaims, Sqrt_TotalClaims, Std_TotalClaims, HighClaim, ClaimSeverityCategory")
        return df

    # -----------------------
    # Remove unnecessary columns (KEEP high-cardinality categorical)
    # -----------------------
    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "\n🗑️ STEP 6: REMOVING UNNECESSARY COLUMNS (identifiers & singletons only)")
        original_cols = df.columns.tolist()
        cols_to_remove = []

        # Identifiers we remove
        id_cols = ['PolicyID', 'TransactionMonth']
        for col in id_cols:
            if col in df.columns:
                cols_to_remove.append(col)
                logger.info(f"  ❌ Removing identifier column: {col}")

        # Single unique value columns
        for col in df.columns:
            if col not in cols_to_remove and df[col].nunique(dropna=False) == 1:
                cols_to_remove.append(col)
                logger.info(f"  ❌ Removing single-unique column: {col}")

        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            logger.info(f"Removed {len(cols_to_remove)} columns")

        logger.info(
            f"Columns reduced from {len(original_cols)} to {len(df.columns)}")
        return df

    # -----------------------
    # Train/test split
    # -----------------------
    def _train_test_split(self, df: pd.DataFrame) -> None:
        """
        Split data into training and testing sets.
        By default stratify using quintiles of Log_TotalClaims (regression) where possible.
        """
        logger.info("\n✂️ STEP 7: TRAIN-TEST SPLIT")
        # define target candidates
        target_candidates = [
            c for c in df.columns if 'Log_TotalClaims' in c] + [self.target_col]
        primary_target = 'Log_TotalClaims' if 'Log_TotalClaims' in df.columns else self.target_col

        X = df[[c for c in df.columns if c !=
                primary_target and not c.endswith('_Claim')]]
        y = df[primary_target]

        # create stratify bin if possible
        try:
            stratify_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
            stratify_col = stratify_bins
            logger.info("Using stratified split on target quintiles")
        except Exception:
            stratify_col = None
            logger.info("Unable to create stratify bins, using random split")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # Also store original target (untransformed) if present
        if self.target_col in df.columns:
            self.y_train_original = df.loc[self.X_train.index, self.target_col]
            self.y_test_original = df.loc[self.X_test.index, self.target_col]
        else:
            self.y_train_original = None
            self.y_test_original = None

        # binary classification target if present
        if 'HighClaim' in df.columns:
            self.y_train_binary = df.loc[self.X_train.index, 'HighClaim']
            self.y_test_binary = df.loc[self.X_test.index, 'HighClaim']

        logger.info(
            f"Training set: {self.X_train.shape[0]:,} rows; Test set: {self.X_test.shape[0]:,} rows")
        logger.info(f"Feature count: {self.X_train.shape[1]}")

    # -----------------------
    # Getters & Save
    # -----------------------
    def get_categorical_cols(self, df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include=['object', 'category']).columns.tolist()

    def get_numerical_cols(self, df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split yet. Call prepare_data() first.")
        return self.X_train.copy(), self.y_train.copy()

    def get_testing_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split yet. Call prepare_data() first.")
        return self.X_test.copy(), self.y_test.copy()

    def get_all_targets(self) -> Dict:
        targets = {
            'regression_log': (getattr(self, 'y_train', None), getattr(self, 'y_test', None)),
            'regression_original': (getattr(self, 'y_train_original', None), getattr(self, 'y_test_original', None)),
        }
        if hasattr(self, 'y_train_binary'):
            targets['binary_classification'] = (
                self.y_train_binary, self.y_test_binary)
        return targets

    def save_prepared_data(self, output_path: str) -> None:
        """Save prepared data to CSV file and save train/test splits."""
        if self.df_prepared is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save full prepared dataset
        self.df_prepared.to_csv(output_path, index=False)
        logger.info(f"💾 Saved prepared data to: {output_path}")

        # Save train/test splits (include original and transformed targets if available)
        train_path = output_path.parent / "train_data.csv"
        test_path = output_path.parent / "test_data.csv"

        train_df = pd.concat([
            self.X_train.reset_index(drop=True),
            pd.DataFrame({'Log_TotalClaims': self.y_train.reset_index(drop=True),
                          self.target_col: self.y_train_original.reset_index(drop=True) if getattr(self, 'y_train_original', None) is not None else pd.Series([np.nan]*len(self.y_train))})
        ], axis=1)

        test_df = pd.concat([
            self.X_test.reset_index(drop=True),
            pd.DataFrame({'Log_TotalClaims': self.y_test.reset_index(drop=True),
                          self.target_col: self.y_test_original.reset_index(drop=True) if getattr(self, 'y_test_original', None) is not None else pd.Series([np.nan]*len(self.y_test))})
        ], axis=1)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"💾 Saved training data to: {train_path}")
        logger.info(f"💾 Saved testing data to: {test_path}")

    def get_data_summary(self) -> Dict:
        """Get summary statistics of the prepared data."""
        if self.df_prepared is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        summary = {
            'original_records': len(self.df),
            'prepared_records': len(self.df_prepared),
            'training_records': len(self.X_train) if self.X_train is not None else 0,
            'testing_records': len(self.X_test) if self.X_test is not None else 0,
            'features_count': self.X_train.shape[1] if self.X_train is not None else 0,
            'numerical_features': len(self.get_numerical_cols(self.df_prepared)),
            'categorical_features': len(self.get_categorical_cols(self.df_prepared)),
            'target_statistics': {
                'original_mean': float(self.df[self.target_col].mean()) if self.target_col in self.df else None,
                'original_std': float(self.df[self.target_col].std()) if self.target_col in self.df else None,
                'log_mean': float(self.df_prepared['Log_TotalClaims'].mean()) if 'Log_TotalClaims' in self.df_prepared.columns else None,
                'log_std': float(self.df_prepared['Log_TotalClaims'].std()) if 'Log_TotalClaims' in self.df_prepared.columns else None
            }
        }
        return summary

    def get_feature_columns(self) -> Dict:
        """Get lists of different feature types."""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        feature_types = {
            'all_features': self.X_train.columns.tolist(),
            'numerical_features': self.get_numerical_cols(self.X_train),
            'categorical_features': self.get_categorical_cols(self.X_train),
            'engineered_features': [col for col in self.X_train.columns if any(keyword in col.lower() for keyword in ['age', 'ratio', 'has', 'is', 'interaction', 'risk', 'power'])]
        }
        return feature_types


# If run as a script, run a quick example using workspace raw data
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Adjust the path to point at your repository workspace raw file
    default_raw = Path("./data/raw/MachineLearningRating_v3.txt")
    default_out = Path("./data/processed/claim_data_prepared.csv")

    preparer = ClaimDataPreparer(
        claim_data_path=str(default_raw),
        target_col='TotalClaims',
        test_size=0.2,
        random_state=42
    )

    df_prepared = preparer.prepare_data(save_path=str(default_out))
    summary = preparer.get_data_summary()

    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Original records: {summary['original_records']:,}")
    print(f"Prepared records: {summary['prepared_records']:,}")
    print(f"Training samples: {summary['training_records']:,}")
    print(f"Testing samples: {summary['testing_records']:,}")
    print(f"Total features: {summary['features_count']}")
    print(f"Numerical features: {summary['numerical_features']}")
    print(f"Categorical features: {summary['categorical_features']}")
    print("\nSample of prepared data:")
    print(df_prepared.head())
