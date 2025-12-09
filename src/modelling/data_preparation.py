"""
Module for preparing claim data for modeling.
"""
from sklearn.preprocessing import LabelEncoder
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
            self.df = pd.read_csv(self.claim_data_path)
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
        self.df_prepared = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.label_encoders = {}

        # Configuration
        self.missing_threshold = 0.5  # Remove columns with >50% missing
        self.max_categories = 50  # Maximum unique values for categorical encoding

        logger.info(
            f"Initialized ClaimDataPreparer with {len(self.df)} records")
        logger.info(f"Target variable: {self.target_col}")
        logger.info(f"Test size: {test_size}")

    def _clean_numeric_string(self, value):
        """Clean numeric strings by extracting numbers from text."""
        if pd.isna(value):
            return np.nan

        # If already numeric, return as is
        if isinstance(value, (int, float, np.number)):
            return float(value)

        # If string, try to extract numeric value
        if isinstance(value, str):
            # Remove any text and keep only numbers and decimal points
            # Handle European number format (comma as decimal)
            value = str(value).strip()

            # Replace comma with dot if it looks like European decimal
            if ',' in value and '.' not in value:
                # Check if comma is likely decimal separator
                if value.count(',') == 1 and len(value.split(',')[-1]) <= 2:
                    value = value.replace(',', '.')

            # Extract numbers including decimals
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    return np.nan

        return np.nan

    def prepare_data(self, save_path: str = None) -> pd.DataFrame:
        """
        Perform all data preparation steps.

        Args:
        save_path: Path to save prepared data (optional)

        Returns:
        Prepared DataFrame
        """
        logger.info("="*60)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("="*60)

        # Step 0: Clean numeric columns first
        df_cleaned = self._clean_numeric_columns()

        # Step 1: Initial data inspection
        self._inspect_data(df_cleaned)

        # Step 2: Handle missing values
        df_missing_handled = self._handle_missing_data(df_cleaned)

        # Step 3: Feature engineering
        df_engineered = self._feature_engineering(df_missing_handled)

        # Step 4: Encode categorical variables
        df_encoded = self._encode_categorical_data(df_engineered)

        # Step 5: Create target transformations
        df_final = self._create_target_variables(df_encoded)

        # Step 6: Remove unnecessary columns
        df_final = self._remove_unnecessary_columns(df_encoded)

        # Step 7: Split data
        self._train_test_split(df_final)

        # Set df_prepared BEFORE saving
        self.df_prepared = df_final  # <-- MOVE THIS LINE UP

        # Step 8: Save if requested
        if save_path:
            self.save_prepared_data(save_path)

        logger.info("✅ Data preparation complete!")

        return df_final

    def _clean_numeric_columns(self) -> pd.DataFrame:
        """Clean numeric columns that may contain text or mixed formats."""
        logger.info("\n🧹 STEP 0: CLEANING NUMERIC COLUMNS")
        logger.info("-"*40)

        df = self.df.copy()

        # Columns that should be numeric
        potential_numeric_cols = [
            'RegistrationYear', 'cubiccapacity', 'kilowatts',
            'CapitalOutstanding', 'SumInsured', 'TotalPremium',
            'TotalClaims', 'ExcessSelected'
        ]

        for col in potential_numeric_cols:
            if col in df.columns:
                original_dtype = df[col].dtype
                original_non_nan = df[col].notna().sum()

                # Clean the column
                df[col] = df[col].apply(self._clean_numeric_string)

                # Count successful conversions
                converted_non_nan = df[col].notna().sum()

                logger.info(f"  🔧 {col}:")
                logger.info(f"     Original dtype: {original_dtype}")
                logger.info(f"     Non-null before: {original_non_nan}")
                logger.info(f"     Non-null after: {converted_non_nan}")
                logger.info(
                    f"     Conversion rate: {converted_non_nan/original_non_nan*100:.1f}%")

        return df

    def _inspect_data(self, df: pd.DataFrame) -> None:
        """Perform initial data inspection."""
        logger.info("\n📊 STEP 1: DATA INSPECTION")
        logger.info("-"*40)

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Check target variable
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in data")

        logger.info(f"\nTarget variable '{self.target_col}':")
        logger.info(f"  Min: {df[self.target_col].min():,.2f}")
        logger.info(f"  Max: {df[self.target_col].max():,.2f}")
        logger.info(f"  Mean: {df[self.target_col].mean():,.2f}")
        logger.info(f"  Std: {df[self.target_col].std():,.2f}")

        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100

        logger.info("\nMissing values analysis:")
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) == 0:
            logger.info("  ✅ No missing values found")
        else:
            for col, count in cols_with_missing.items():
                pct = missing_percentage[col]
                logger.info(f"  ⚠️  {col}: {count} missing ({pct:.2f}%)")

        # Check data types
        logger.info("\nData types:")
        for dtype in df.dtypes.unique():
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            logger.info(f"  {dtype}: {len(cols)} columns")

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Strategy:
        - Remove columns with >50% missing values
        - Impute numerical columns with median
        - Impute categorical columns with mode
        """
        logger.info("\n🔧 STEP 2: HANDLING MISSING DATA")
        logger.info("-"*40)

        original_shape = df.shape

        # Step 1: Remove columns with excessive missing values
        missing_percentage = df.isnull().sum() / len(df)
        cols_to_drop = missing_percentage[missing_percentage >
                                          self.missing_threshold].index.tolist()

        if cols_to_drop:
            logger.info(
                f"Removing columns with >{self.missing_threshold*100:.0f}% missing values:")
            for col in cols_to_drop:
                logger.info(
                    f"  ❌ {col}: {missing_percentage[col]*100:.1f}% missing")
            df = df.drop(columns=cols_to_drop)

        # Step 2: Identify numerical and categorical columns
        self.numerical_columns = df.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(
            include=['object']).columns.tolist()

        # Remove target from numerical columns if present
        if self.target_col in self.numerical_columns:
            self.numerical_columns.remove(self.target_col)

        logger.info(f"Numerical columns: {len(self.numerical_columns)}")
        logger.info(f"Categorical columns: {len(self.categorical_columns)}")

        # Step 3: Impute numerical columns
        if self.numerical_columns:
            logger.info("\nImputing numerical columns with median:")
            for col in self.numerical_columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(
                        f"  🔧 {col}: {missing_count} values filled with {median_val:.2f}")

        # Step 4: Impute categorical columns
        if self.categorical_columns:
            logger.info("\nImputing categorical columns with mode:")
            for col in self.categorical_columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    mode_val = df[col].mode(
                    )[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    logger.info(
                        f"  🔧 {col}: {missing_count} values filled with '{mode_val}'")

        logger.info(
            f"\nMissing values after handling: {df.isnull().sum().sum()}")
        logger.info(f"Shape changed from {original_shape} to {df.shape}")

        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features that might be relevant to TotalClaims.
        """
        logger.info("\n⚙️ STEP 3: FEATURE ENGINEERING")
        logger.info("-"*40)

        original_cols = df.columns.tolist()

        # 1. Temporal features from TransactionMonth
        if 'TransactionMonth' in df.columns:
            logger.info("Creating temporal features...")
            try:
                df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
                df['TransactionYear'] = df['TransactionMonth'].dt.year
                df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
                df['TransactionQuarter'] = df['TransactionMonth'].dt.quarter
                df['IsYearEnd'] = df['TransactionMonthNum'].isin(
                    [11, 12]).astype(int)
            except Exception as e:
                logger.warning(f"Could not parse TransactionMonth: {str(e)}")
                # Create simple year feature if parsing fails
                if df['TransactionMonth'].dtype == 'object':
                    # Try to extract year from string
                    def extract_year(x):
                        if isinstance(x, str):
                            year_match = re.search(r'(\d{4})', x)
                            if year_match:
                                return int(year_match.group(1))
                        return np.nan

                    df['TransactionYear'] = df['TransactionMonth'].apply(
                        extract_year)
                    df['TransactionYear'] = df['TransactionYear'].fillna(
                        df['TransactionYear'].median())

        # 2. Vehicle features
        if 'RegistrationYear' in df.columns:
            logger.info("Creating vehicle age features...")
            # Use max of transaction year or current year
            if 'TransactionYear' in df.columns:
                reference_year = df['TransactionYear'].max()
            else:
                reference_year = pd.Timestamp.now().year

            df['VehicleAge'] = reference_year - df['RegistrationYear']
            df['VehicleAge'] = df['VehicleAge'].clip(lower=0, upper=50)
            df['IsOldVehicle'] = (df['VehicleAge'] > 10).astype(int)
            df['IsNewVehicle'] = (df['VehicleAge'] < 3).astype(int)

        # 3. Engine and power features
        if all(col in df.columns for col in ['cubiccapacity', 'kilowatts']):
            logger.info("Creating engine power features...")
            df['PowerToCapacityRatio'] = df['kilowatts'] / \
                (df['cubiccapacity'].replace(0, np.nan) + 1)
            df['HasHighPower'] = (
                df['PowerToCapacityRatio'] > df['PowerToCapacityRatio'].median()).astype(int)

        # 4. Financial and risk features
        if all(col in df.columns for col in ['SumInsured', 'TotalPremium']):
            logger.info("Creating premium risk features...")
            df['PremiumToSumInsuredRatio'] = df['TotalPremium'] / \
                (df['SumInsured'].replace(0, np.nan) + 1)

            # Handle infinite values
            df['PremiumToSumInsuredRatio'] = df['PremiumToSumInsuredRatio'].replace(
                [np.inf, -np.inf], np.nan)
            median_ratio = df['PremiumToSumInsuredRatio'].median()
            df['PremiumToSumInsuredRatio'] = df['PremiumToSumInsuredRatio'].fillna(
                median_ratio)

            df['IsHighRiskPremium'] = (df['PremiumToSumInsuredRatio'] >
                                       df['PremiumToSumInsuredRatio'].quantile(0.75)).astype(int)

        # 5. Excess features - handle string values
        if 'ExcessSelected' in df.columns:
            logger.info("Creating excess features...")
            # Check if ExcessSelected is numeric
            if pd.api.types.is_numeric_dtype(df['ExcessSelected']):
                df['HasHighExcess'] = (
                    df['ExcessSelected'] > df['ExcessSelected'].median()).astype(int)
            else:
                # Extract numeric value from string
                def extract_excess_value(x):
                    if pd.isna(x):
                        return np.nan
                    if isinstance(x, (int, float)):
                        return float(x)

                    # Extract numbers from string
                    numbers = re.findall(r'\d+', str(x))
                    if numbers:
                        # Take the last number (likely the excess amount)
                        return float(numbers[-1])
                    return np.nan

                df['ExcessAmount'] = df['ExcessSelected'].apply(
                    extract_excess_value)
                df['ExcessAmount'] = df['ExcessAmount'].fillna(
                    df['ExcessAmount'].median())
                df['HasHighExcess'] = (
                    df['ExcessAmount'] > df['ExcessAmount'].median()).astype(int)

        # 6. Security features
        security_cols = ['AlarmImmobiliser', 'TrackingDevice']
        security_present = [col for col in security_cols if col in df.columns]

        if security_present:
            logger.info("Creating security features...")
            df['HasSecurity'] = 0
            for col in security_present:
                # Convert to binary: has security or not
                df['HasSecurity'] += df[col].notna().astype(int)
            df['HasSecurity'] = df['HasSecurity'].clip(upper=2)
            df['HasAdvancedSecurity'] = (df['HasSecurity'] == 2).astype(int)

        # 7. Geographic risk features
        if 'MainCrestaZone' in df.columns:
            logger.info("Creating geographic risk features...")
            # Simple risk indicator based on zone
            df['IsHighRiskZone'] = df['MainCrestaZone'].str.contains(
                'A|1|High', case=False, na=False).astype(int)

        # 8. Vehicle make/model features
        if 'make' in df.columns:
            logger.info("Creating vehicle make features...")
            # Top makes
            top_makes = df['make'].value_counts().head(10).index.tolist()
            df['IsTopMake'] = df['make'].isin(top_makes).astype(int)

        # 9. Coverage features
        if 'CoverType' in df.columns:
            logger.info("Creating coverage features...")
            df['IsComprehensive'] = df['CoverType'].str.contains(
                'Comprehensive', case=False, na=False).astype(int)

        # 10. Interaction features (only if base columns exist)
        if all(col in df.columns for col in ['VehicleAge', 'SumInsured']):
            df['AgeValueInteraction'] = df['VehicleAge'] * df['SumInsured']

        # 11. Capital features
        if 'CapitalOutstanding' in df.columns:
            logger.info("Creating capital features...")
            df['HasCapitalOutstanding'] = (
                df['CapitalOutstanding'] > 0).astype(int)
            df['CapitalToSumInsuredRatio'] = df['CapitalOutstanding'] / \
                (df['SumInsured'].replace(0, np.nan) + 1)
            df['CapitalToSumInsuredRatio'] = df['CapitalToSumInsuredRatio'].replace(
                [np.inf, -np.inf], np.nan)
            df['CapitalToSumInsuredRatio'] = df['CapitalToSumInsuredRatio'].fillna(
                0)

        new_cols = [col for col in df.columns if col not in original_cols]
        logger.info(f"Created {len(new_cols)} new features:")
        for col in new_cols[:10]:  # Show first 10
            logger.info(f"  ✅ {col}")
        if len(new_cols) > 10:
            logger.info(f"  ... and {len(new_cols) - 10} more")

        return df

    def _encode_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for modeling.
        """
        logger.info("\n🔢 STEP 4: ENCODING CATEGORICAL DATA")
        logger.info("-"*40)

        df_encoded = df.copy()

        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(
            include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return df_encoded

        logger.info(f"Found {len(categorical_cols)} categorical columns")

        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()

            # Skip if too many unique values
            if unique_count > 100:
                logger.info(
                    f"  ⚠️  {col}: {unique_count} unique values - dropping (likely ID/free text)")
                df_encoded = df_encoded.drop(columns=[col])
                continue

            # Label encoding for binary variables
            if unique_count == 2:
                logger.info(f"  🔤 {col}: Binary variable - label encoding")
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le

            # For moderate cardinality, keep as is for one-hot encoding in modeling
            elif unique_count <= self.max_categories:
                logger.info(
                    f"  🔤 {col}: {unique_count} categories - will one-hot encode in modeling")
                # Will be handled during modeling

            # Frequency encoding for high cardinality
            else:
                logger.info(
                    f"  🔤 {col}: {unique_count} categories - frequency encoding")
                freq_map = df_encoded[col].value_counts(
                    normalize=True).to_dict()
                df_encoded[f'{col}_FreqEncoded'] = df_encoded[col].map(
                    freq_map)
                df_encoded = df_encoded.drop(columns=[col])

        # Update categorical columns list
        self.categorical_columns = df_encoded.select_dtypes(
            include=['object', 'category']).columns.tolist()

        logger.info(
            f"\nRemaining categorical columns after encoding: {len(self.categorical_columns)}")

        return df_encoded

    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transformed versions of the target variable.
        """
        logger.info("\n🎯 STEP 5: CREATING TARGET VARIABLES")
        logger.info("-"*40)

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in prepared data")

        # Ensure target is numeric
        if not pd.api.types.is_numeric_dtype(df[self.target_col]):
            df[self.target_col] = pd.to_numeric(
                df[self.target_col], errors='coerce')

        # Remove any zeros or negative values
        df = df[df[self.target_col] > 0]

        # 1. Log transformation
        df['Log_TotalClaims'] = np.log1p(df[self.target_col])

        # 2. Square root transformation
        df['Sqrt_TotalClaims'] = np.sqrt(df[self.target_col])

        # 3. Standardized version
        mean_val = df[self.target_col].mean()
        std_val = df[self.target_col].std()
        df['Std_TotalClaims'] = (
            df[self.target_col] - mean_val) / (std_val + 1e-10)

        # 4. Binary classification target
        median_claim = df[self.target_col].median()
        df['HighClaim'] = (df[self.target_col] > median_claim).astype(int)

        # 5. Claim severity categories
        quantiles = df[self.target_col].quantile([0.33, 0.66])
        if not quantiles.empty:
            df['ClaimSeverityCategory'] = pd.cut(
                df[self.target_col],
                bins=[0, quantiles.iloc[0], quantiles.iloc[1], np.inf],
                labels=['Low', 'Medium', 'High']
            )

        logger.info(f"Created target transformations:")
        logger.info(f"  ✅ Log_TotalClaims: Log(1 + claim_amount)")
        logger.info(f"  ✅ HighClaim: Binary (1 if claim > median)")

        logger.info(f"\nTarget distribution after transformations:")
        logger.info(f"  Original mean: R{df[self.target_col].mean():,.2f}")
        logger.info(f"  Log mean: {df['Log_TotalClaims'].mean():.2f}")
        logger.info(
            f"  High claims percentage: {df['HighClaim'].mean()*100:.1f}%")

        return df

    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that are not useful for modeling.
        """
        logger.info("\n🗑️  STEP 6: REMOVING UNNECESSARY COLUMNS")
        logger.info("-"*40)

        original_cols = df.columns.tolist()
        cols_to_remove = []

        # 1. Identifier columns
        id_cols = ['PolicyID', 'TransactionMonth']
        for col in id_cols:
            if col in df.columns:
                cols_to_remove.append(col)
                logger.info(f"  ❌ {col}: Identifier column")

        # 2. Columns with single unique value
        for col in df.columns:
            if col not in cols_to_remove and df[col].nunique() == 1:
                cols_to_remove.append(col)
                logger.info(f"  ❌ {col}: Single unique value")

        # 3. High cardinality text columns
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in text_cols:
            if col not in cols_to_remove and df[col].nunique() > 20:
                cols_to_remove.append(col)
                logger.info(
                    f"  ❌ {col}: High cardinality ({df[col].nunique()} unique values)")

        # Remove the columns
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            logger.info(f"\nRemoved {len(cols_to_remove)} columns")

        logger.info(
            f"Columns reduced from {len(original_cols)} to {len(df.columns)}")

        return df

    def _train_test_split(self, df: pd.DataFrame) -> None:
        """
        Split data into training and testing sets.
        """
        logger.info("\n✂️  STEP 7: TRAIN-TEST SPLIT")
        logger.info("-"*40)

        # Define features and target
        target_cols = [col for col in df.columns if 'TotalClaims' in col or col in [
            'HighClaim', 'ClaimSeverityCategory']]
        feature_cols = [col for col in df.columns if col not in target_cols]

        # Use Log_TotalClaims as primary regression target
        if 'Log_TotalClaims' in df.columns:
            primary_target = 'Log_TotalClaims'
        else:
            primary_target = self.target_col

        X = df[feature_cols]
        y = df[primary_target]

        # Create bins for stratification
        try:
            y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
            stratify_col = y_binned
            logger.info(
                "Using stratified split based on target distribution...")
        except:
            logger.info("Using random split...")
            stratify_col = None

        # Perform the split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # Also store the other targets
        self.y_train_original = df.loc[self.X_train.index, self.target_col]
        self.y_test_original = df.loc[self.X_test.index, self.target_col]

        if 'HighClaim' in df.columns:
            self.y_train_binary = df.loc[self.X_train.index, 'HighClaim']
            self.y_test_binary = df.loc[self.X_test.index, 'HighClaim']

        logger.info(
            f"Training set: {self.X_train.shape[0]:,} samples ({100*(1-self.test_size):.0f}%)")
        logger.info(
            f"Test set: {self.X_test.shape[0]:,} samples ({100*self.test_size:.0f}%)")
        logger.info(f"Number of features: {self.X_train.shape[1]}")

        # Check distribution
        logger.info(f"\nTarget distribution in train set:")
        logger.info(f"  Mean: {self.y_train.mean():.2f}")
        logger.info(f"  Std: {self.y_train.std():.2f}")

        logger.info(f"\nTarget distribution in test set:")
        logger.info(f"  Mean: {self.y_test.mean():.2f}")
        logger.info(f"  Std: {self.y_test.std():.2f}")

    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split yet. Call prepare_data() first.")
        return self.X_train, self.y_train

    def get_testing_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get testing data."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split yet. Call prepare_data() first.")
        return self.X_test, self.y_test

    def get_all_targets(self) -> Dict:
        """Get all target variables for different modeling tasks."""
        targets = {
            'regression_log': (self.y_train, self.y_test),
            'regression_original': (self.y_train_original, self.y_test_original),
        }

        if hasattr(self, 'y_train_binary'):
            targets['binary_classification'] = (
                self.y_train_binary, self.y_test_binary)

        return targets

    def save_prepared_data(self, output_path: str) -> None:
        """Save prepared data to CSV file."""
        if self.df_prepared is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save full prepared dataset
        self.df_prepared.to_csv(output_path, index=False)
        logger.info(f"💾 Saved prepared data to: {output_path}")

        # Save train/test splits
        train_path = output_path.parent / 'train_data.csv'
        test_path = output_path.parent / 'test_data.csv'

        train_df = pd.concat([self.X_train, pd.DataFrame(
            {'Log_TotalClaims': self.y_train})], axis=1)
        test_df = pd.concat([self.X_test, pd.DataFrame(
            {'Log_TotalClaims': self.y_test})], axis=1)

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
            'numerical_features': len(self.numerical_columns) if self.numerical_columns else 0,
            'categorical_features': len(self.categorical_columns) if self.categorical_columns else 0,
            'target_statistics': {
                'original_mean': float(self.df[self.target_col].mean()),
                'original_std': float(self.df[self.target_col].std()),
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
            'numerical_features': self.numerical_columns if self.numerical_columns else [],
            'categorical_features': self.categorical_columns if self.categorical_columns else [],
            'engineered_features': [col for col in self.X_train.columns
                                    if any(keyword in col.lower() for keyword in
                                           ['age', 'ratio', 'has', 'is', 'interaction', 'risk', 'power'])]
        }

        return feature_types


# Main execution block
if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Create preparer
    preparer = ClaimDataPreparer(
        claim_data_path=r"D:\Python\Week-3\Insurance-Analytics-Week-3-\data\processed\claim_policies_subset.csv",
        target_col='TotalClaims',
        test_size=0.2,
        random_state=42
    )

    # Run complete preparation pipeline
    try:
        df_prepared = preparer.prepare_data(
            save_path=r"D:\Python\Week-3\Insurance-Analytics-Week-3-\data\processed\claim_data_prepared.csv"
        )

        # Get summary
        summary = preparer.get_data_summary()

        print("\n" + "="*60)
        print("DATA PREPARATION SUMMARY")
        print("="*60)
        print(f"Original records: {summary['original_records']:,}")
        print(f"Prepared records: {summary['prepared_records']:,}")
        print(f"Training samples: {summary['training_records']:,}")
        print(f"Testing samples: {summary['testing_records']:,}")
        print(f"Total features: {summary['features_count']}")
        print(f"Numerical features: {summary['numerical_features']}")
        print(f"Categorical features: {summary['categorical_features']}")
        print("\nTarget Statistics:")
        print(
            f"  Original mean: R{summary['target_statistics']['original_mean']:,.2f}")
        print(
            f"  Original std: R{summary['target_statistics']['original_std']:,.2f}")
        if summary['target_statistics']['log_mean']:
            print(
                f"  Log mean: {summary['target_statistics']['log_mean']:.2f}")
            print(f"  Log std: {summary['target_statistics']['log_std']:.2f}")

        # Show sample of prepared data
        print("\nSample of prepared data:")
        print(df_prepared.head())

        print("\n" + "="*60)
        print("✅ Data preparation complete! Ready for modeling.")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error in data preparation: {str(e)}")
        import traceback
        traceback.print_exc()
