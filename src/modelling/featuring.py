"""
Module for advanced feature engineering and preprocessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering, encoding, scaling, and preprocessing for claim severity modeling.
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.categorical_features = None
        self.numerical_features = None
        self.preprocessor = None
        self.feature_names = None
        self.training_columns_ = None

        # Configuration
        self.onehot_threshold = 20  # Max categories for one-hot encoding

        logger.info("Initialized FeatureEngineer")

    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify categorical and numerical features.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        logger.info("Identifying feature types...")

        # Numerical features
        numerical_features = X.select_dtypes(
            include=[np.number]).columns.tolist()

        # Categorical features
        categorical_features = X.select_dtypes(
            include=['object', 'category']).columns.tolist()

        # Also consider low-cardinality integer columns as categorical
        for col in X.select_dtypes(include=['int64']).columns:
            if col not in numerical_features and X[col].nunique() < 10:
                categorical_features.append(col)
                if col in numerical_features:
                    numerical_features.remove(col)

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        logger.info(f"Identified {len(numerical_features)} numerical features")
        logger.info(
            f"Identified {len(categorical_features)} categorical features")

        # Log some examples
        if numerical_features:
            logger.info(
                f"  Numerical: {numerical_features[:5]}{'...' if len(numerical_features) > 5 else ''}")
        if categorical_features:
            logger.info(
                f"  Categorical: {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}")

        return categorical_features, numerical_features

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for features.

        Returns:
            ColumnTransformer for preprocessing
        """
        if self.categorical_features is None or self.numerical_features is None:
            raise ValueError(
                "Feature types not identified. Call identify_feature_types() first.")

        logger.info("Creating preprocessing pipeline...")

        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first'  # Drop first category to avoid dummy variable trap
            ))
        ])

        # Column transformer
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])

        logger.info(
            f"  One-hot encoding for {len(self.categorical_features)} categorical features")
        logger.info(
            f"  Standard scaling for {len(self.numerical_features)} numerical features")
        logger.info("✅ Preprocessing pipeline created")

        return self.preprocessor

    def fit_transform_features(self,
                               X_train: pd.DataFrame,
                               X_test: pd.DataFrame = None) -> Tuple:
        """
        Fit preprocessing on training data and transform both train and test.
        Ensures test data has same columns as training data.

        Args:
            X_train: Training features
            X_test: Test features (optional)

        Returns:
            Tuple of (X_train_transformed, X_test_transformed or None)
        """
        logger.info("Fitting and transforming features...")

        # Store training columns
        self.training_columns_ = X_train.columns.tolist()
        logger.info(f"Training data has {len(self.training_columns_)} columns")

        # Identify feature types
        self.identify_feature_types(X_train)

        # Create preprocessing pipeline
        self.create_preprocessing_pipeline()

        # Fit and transform training data
        logger.info("Fitting preprocessor on training data...")
        X_train_transformed = self.preprocessor.fit_transform(X_train)

        # Get feature names
        self._get_feature_names(X_train)

        # Transform test data if provided
        X_test_transformed = None
        if X_test is not None:
            logger.info("Transforming test data...")

            # Check if test data has all required columns
            missing_cols = set(self.training_columns_) - set(X_test.columns)

            if missing_cols:
                logger.warning(f"Test data missing {len(missing_cols)} columns. "
                               f"Adding them with default values.")

                # Add missing columns to test data
                for col in missing_cols:
                    if col in self.numerical_features:
                        # For numerical columns, use median from training
                        if col in X_train.columns:
                            default_value = X_train[col].median()
                        else:
                            default_value = 0
                        X_test[col] = default_value
                        logger.debug(
                            f"  Added numerical column '{col}' with value: {default_value}")
                    elif col in self.categorical_features:
                        # For categorical columns, use mode from training
                        if col in X_train.columns:
                            mode_values = X_train[col].mode()
                            default_value = mode_values[0] if not mode_values.empty else 'Unknown'
                        else:
                            default_value = 'Unknown'
                        X_test[col] = default_value
                        logger.debug(
                            f"  Added categorical column '{col}' with value: '{default_value}'")
                    else:
                        # For other columns, use 0 or appropriate default
                        X_test[col] = 0
                        logger.debug(
                            f"  Added column '{col}' with default value: 0")

            # Also check for extra columns in test data (not in training)
            extra_cols = set(X_test.columns) - set(self.training_columns_)
            if extra_cols:
                logger.warning(f"Test data has {len(extra_cols)} extra columns. "
                               f"Dropping them to match training data.")
                X_test = X_test[self.training_columns_]

            # Ensure column order matches training data
            if not all(X_test.columns == self.training_columns_):
                logger.debug("Reordering test columns to match training data")
                X_test = X_test[self.training_columns_]

            # Now transform the aligned test data
            try:
                X_test_transformed = self.preprocessor.transform(X_test)
                logger.info(
                    f"  Test features shape: {X_test_transformed.shape}")
            except Exception as e:
                logger.error(f"Error transforming test data: {str(e)}")
                logger.error(f"Training columns: {self.training_columns_}")
                logger.error(f"Test columns: {X_test.columns.tolist()}")
                raise

        logger.info(f"  Training features shape: {X_train_transformed.shape}")
        logger.info(
            f"  Total features after preprocessing: {X_train_transformed.shape[1]}")

        return X_train_transformed, X_test_transformed

    def _align_columns(self, X: pd.DataFrame, reference_columns: list = None) -> pd.DataFrame:
        """
        Align DataFrame columns to match reference columns.

        Args:
            X: DataFrame to align
            reference_columns: Reference columns (default: training_columns_)

        Returns:
            Aligned DataFrame
        """
        if reference_columns is None:
            if not hasattr(self, 'training_columns_'):
                raise ValueError(
                    "No reference columns provided and training_columns_ not available")
            reference_columns = self.training_columns_

        # Create a copy
        X_aligned = X.copy()

        # Add missing columns
        missing_cols = set(reference_columns) - set(X_aligned.columns)
        for col in missing_cols:
            if col in self.numerical_features:
                X_aligned[col] = 0
            elif col in self.categorical_features:
                X_aligned[col] = 'Unknown'
            else:
                X_aligned[col] = 0

        # Reorder columns
        X_aligned = X_aligned[reference_columns]

        return X_aligned

    def transform_new_data(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted preprocessor.

        Args:
            X_new: New data to transform

        Returns:
            Transformed features
        """
        if not hasattr(self, 'training_columns_'):
            raise ValueError(
                "Preprocessor has not been fitted yet. Call fit_transform_features first.")

        if not hasattr(self, 'preprocessor'):
            raise ValueError(
                "Preprocessor has not been created. Call create_preprocessing_pipeline first.")

        # Align columns to match training
        X_new_aligned = self._align_columns(X_new)

        # Transform
        return self.preprocessor.transform(X_new_aligned)

    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after preprocessing."""
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not created. Call create_preprocessing_pipeline() first.")

        feature_names = []

        # Get numerical feature names
        feature_names.extend(self.numerical_features)

        # Get categorical feature names from OneHotEncoder
        categorical_transformer = self.preprocessor.named_transformers_['cat']
        if hasattr(categorical_transformer, 'named_steps'):
            onehot = categorical_transformer.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                # Get original categorical columns that were transformed
                cat_features_in = self.categorical_features
                cat_features_out = onehot.get_feature_names_out(
                    cat_features_in)
                feature_names.extend(cat_features_out)

        self.feature_names = feature_names

        # Log some information
        logger.info(f"Generated {len(feature_names)} feature names")
        logger.info(f"  Numerical: {len(self.numerical_features)}")
        logger.info(
            f"  Categorical (encoded): {len(feature_names) - len(self.numerical_features)}")

        return feature_names

    def create_additional_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Create additional interaction and polynomial features.

        Args:
            X: Input features
            y: Target variable (optional, for target encoding)

        Returns:
            DataFrame with additional features
        """
        logger.info("Creating additional advanced features...")

        X_enhanced = X.copy()
        original_cols = X_enhanced.columns.tolist()

        # Ensure we have numerical columns
        if self.numerical_features is None:
            self.identify_feature_types(X)

        # 1. Polynomial features for important numerical columns
        important_num_cols = ['VehicleAge', 'SumInsured',
                              'TotalPremium', 'cubiccapacity']
        available_num_cols = [
            col for col in important_num_cols if col in X_enhanced.columns]

        for col in available_num_cols:
            try:
                # Skip if column has too many zeros or constant values
                if X_enhanced[col].nunique() > 1:
                    X_enhanced[f'{col}_squared'] = X_enhanced[col] ** 2
                    X_enhanced[f'{col}_cubed'] = X_enhanced[col] ** 3
                    # Use log1p to handle zeros
                    X_enhanced[f'log_{col}'] = np.log1p(
                        np.abs(X_enhanced[col]))
            except Exception as e:
                logger.warning(
                    f"Could not create polynomial features for {col}: {str(e)}")

        # 2. Interaction features (with error handling)
        try:
            if 'VehicleAge' in X_enhanced.columns and 'SumInsured' in X_enhanced.columns:
                X_enhanced['Age_Value_Interaction'] = X_enhanced['VehicleAge'] * \
                    X_enhanced['SumInsured']
                # Avoid division by zero
                denominator = X_enhanced['SumInsured'].replace(0, np.nan)
                X_enhanced['Age_Value_Ratio'] = X_enhanced['VehicleAge'] / denominator
                X_enhanced['Age_Value_Ratio'] = X_enhanced['Age_Value_Ratio'].fillna(
                    0)
        except Exception as e:
            logger.warning(f"Could not create interaction features: {str(e)}")

        try:
            if 'TotalPremium' in X_enhanced.columns and 'SumInsured' in X_enhanced.columns:
                denominator = X_enhanced['SumInsured'].replace(0, np.nan)
                X_enhanced['Premium_Per_Value'] = X_enhanced['TotalPremium'] / denominator
                X_enhanced['Premium_Per_Value'] = X_enhanced['Premium_Per_Value'].fillna(
                    0)
        except:
            pass

        # 3. Risk score features
        risk_factors = []
        if 'VehicleAge' in X_enhanced.columns:
            risk_factors.append(
                (X_enhanced['VehicleAge'] > 10).astype(int))  # Old vehicles

        if 'cubiccapacity' in X_enhanced.columns:
            median_cc = X_enhanced['cubiccapacity'].median()
            risk_factors.append(
                # Large engine
                (X_enhanced['cubiccapacity'] > median_cc).astype(int))

        if 'HasHighPower' in X_enhanced.columns:
            risk_factors.append(X_enhanced['HasHighPower'])

        if risk_factors:
            X_enhanced['Risk_Score'] = sum(risk_factors) / len(risk_factors)

        # 4. Target encoding if y is provided
        if y is not None and len(self.categorical_features) > 0:
            logger.info("Creating target-encoded features...")
            # Limit to top 3 categorical
            for col in self.categorical_features[:3]:
                if col in X_enhanced.columns and X_enhanced[col].nunique() > 1:
                    try:
                        # Calculate mean target per category
                        target_df = pd.DataFrame(
                            {'feature': X_enhanced[col], 'target': y})
                        target_means = target_df.groupby(
                            'feature')['target'].mean().to_dict()

                        X_enhanced[f'{col}_TargetEncoded'] = X_enhanced[col].map(
                            target_means)
                        # Fill NaN with overall mean
                        overall_mean = X_enhanced[f'{col}_TargetEncoded'].mean(
                        )
                        X_enhanced[f'{col}_TargetEncoded'] = X_enhanced[f'{col}_TargetEncoded'].fillna(
                            overall_mean)
                    except Exception as e:
                        logger.warning(
                            f"Could not create target encoding for {col}: {str(e)}")

        # 5. Binning numerical features (with robust binning)
        try:
            if 'VehicleAge' in X_enhanced.columns:
                # Use custom bins instead of pd.cut to avoid edge cases
                bins = [0, 3, 7, 12, 20, 100]
                labels = ['New', 'Young', 'Middle', 'Old', 'VeryOld']

                # Handle edge cases
                X_enhanced['VehicleAge'] = X_enhanced['VehicleAge'].clip(
                    0, 100)
                X_enhanced['VehicleAge_Binned'] = pd.cut(
                    X_enhanced['VehicleAge'],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )
        except Exception as e:
            logger.warning(f"Could not bin VehicleAge: {str(e)}")

        try:
            if 'SumInsured' in X_enhanced.columns:
                # Use pd.qcut with duplicates='drop' to handle duplicate bin edges
                try:
                    X_enhanced['SumInsured_Binned'] = pd.qcut(
                        X_enhanced['SumInsured'],
                        q=5,
                        labels=['VeryLow', 'Low',
                                'Medium', 'High', 'VeryHigh'],
                        duplicates='drop'
                    )
                except:
                    # If qcut fails, use manual percentiles
                    percentiles = X_enhanced['SumInsured'].quantile(
                        [0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    X_enhanced['SumInsured_Binned'] = pd.cut(
                        X_enhanced['SumInsured'],
                        bins=percentiles.unique(),  # Use unique percentiles only
                        labels=['VeryLow', 'Low',
                                'Medium', 'High', 'VeryHigh'],
                        include_lowest=True
                    )
        except Exception as e:
            logger.warning(f"Could not bin SumInsured: {str(e)}")

        # 6. Create flag for missing values in original data
        for col in original_cols:
            if X_enhanced[col].isnull().any():
                X_enhanced[f'{col}_Missing'] = X_enhanced[col].isnull().astype(
                    int)

        new_cols = [
            col for col in X_enhanced.columns if col not in original_cols]
        logger.info(f"Created {len(new_cols)} additional features")

        if new_cols:
            logger.info(
                f"New features: {new_cols[:10]}{'...' if len(new_cols) > 10 else ''}")

        return X_enhanced

    def select_features_using_correlation(self,
                                          X: pd.DataFrame,
                                          y: pd.Series,
                                          threshold: float = 0.1) -> pd.DataFrame:
        """
        Select features based on correlation with target.

        Args:
            X: Feature DataFrame
            y: Target variable
            threshold: Absolute correlation threshold

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting features with |correlation| > {threshold}...")

        # Ensure X is numeric for correlation calculation
        X_numeric = X.select_dtypes(include=[np.number]).copy()

        if len(X_numeric.columns) == 0:
            logger.warning("No numerical features for correlation selection")
            return X

        # Calculate correlations
        correlations = []
        for col in X_numeric.columns:
            try:
                # Drop NaN for correlation calculation
                mask = X_numeric[col].notna() & y.notna()
                if mask.sum() > 10:  # Need enough samples
                    x_vals = X_numeric.loc[mask, col]
                    y_vals = y[mask]
                    corr = np.corrcoef(x_vals, y_vals)[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, abs(corr)))
            except:
                continue

        if not correlations:
            logger.warning("Could not calculate correlations")
            return X

        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)

        # Select features above threshold
        selected_numeric = [feat for feat,
                            corr in correlations if corr > threshold]

        if not selected_numeric:
            logger.warning(
                f"No features above threshold {threshold}. Selecting top 10.")
            selected_numeric = [feat for feat, _ in correlations[:10]]

        # Include categorical features
        selected_features = selected_numeric.copy()
        if self.categorical_features:
            selected_features.extend(self.categorical_features)

        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top correlated features:")
        for feat, corr in correlations[:10]:
            logger.info(f"  {feat}: {corr:.3f}")

        return X[selected_features]

    def select_features_using_variance(self,
                                       X: pd.DataFrame,
                                       threshold: float = 0.01) -> pd.DataFrame:
        """
        Select features based on variance threshold.

        Args:
            X: Feature DataFrame
            threshold: Minimum variance threshold

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting features with variance > {threshold}...")

        # Select only numerical features
        X_numeric = X.select_dtypes(include=[np.number]).copy()

        if len(X_numeric.columns) == 0:
            return X

        # Calculate variances
        variances = X_numeric.var()

        # Select features above variance threshold
        high_variance_features = variances[variances >
                                           threshold].index.tolist()

        # Include categorical features
        selected_features = high_variance_features.copy()
        if self.categorical_features:
            selected_features.extend(self.categorical_features)

        logger.info(
            f"Selected {len(selected_features)} features (variance > {threshold})")
        logger.info(
            f"Dropped {len(X_numeric.columns) - len(high_variance_features)} low-variance features")

        return X[selected_features]

    def calculate_feature_importance(self,
                                     model: Any,
                                     X_transformed: np.ndarray) -> pd.DataFrame:
        """
        Calculate feature importance from trained model.

        Args:
            model: Trained model with feature_importances_ attribute
            X_transformed: Transformed feature matrix

        Returns:
            DataFrame with feature importances
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(
                "Model does not have feature_importances_ attribute")
            return pd.DataFrame()

        if self.feature_names is None:
            logger.warning("Feature names not available")
            return pd.DataFrame()

        # Ensure feature names match importance array length
        n_features = len(model.feature_importances_)
        if n_features != len(self.feature_names):
            logger.warning(f"Feature count mismatch: model has {n_features}, "
                           f"feature names has {len(self.feature_names)}")
            # Use generic names if mismatch
            feature_names_for_importance = [
                f"feature_{i}" for i in range(n_features)]
        else:
            feature_names_for_importance = self.feature_names

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names_for_importance,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(
            f"Calculated feature importance for {len(importance_df)} features")
        logger.info(f"Top 5 features:")
        for idx, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def save_preprocessor(self, filepath: str) -> None:
        """Save preprocessor to file."""
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not created. Call create_preprocessing_pipeline() first.")

        import pickle

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.preprocessor, f)

        logger.info(f"💾 Saved preprocessor to: {filepath}")

    def load_preprocessor(self, filepath: str) -> None:
        """Load preprocessor from file."""
        import pickle

        with open(filepath, 'rb') as f:
            self.preprocessor = pickle.load(f)

        logger.info(f"📂 Loaded preprocessor from: {filepath}")


# Main execution block for testing
if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("="*60)
    print("FEATURE ENGINEERING MODULE TEST")
    print("="*60)

    # Load prepared data
    data_path = r"D:\Python\Week-3\Insurance-Analytics-Week-3-\data\processed\train_data.csv"

    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        print(f"Loaded data with shape: {df.shape}")

        # Separate features and target
        X = df.drop(columns=['Log_TotalClaims'], errors='ignore')
        y = df['Log_TotalClaims'] if 'Log_TotalClaims' in df.columns else None

        # Initialize feature engineer
        feature_engineer = FeatureEngineer()

        # Identify feature types
        cat_features, num_features = feature_engineer.identify_feature_types(X)

        # Create additional features (with y if available)
        if y is not None:
            X_enhanced = feature_engineer.create_additional_features(X, y)
        else:
            X_enhanced = feature_engineer.create_additional_features(X)

        print(f"\nEnhanced features shape: {X_enhanced.shape}")

        # Select features using correlation if y is available
        if y is not None:
            X_selected = feature_engineer.select_features_using_correlation(
                X_enhanced, y, threshold=0.05)
            print(f"Selected features shape: {X_selected.shape}")

            # Also select by variance
            X_variance_selected = feature_engineer.select_features_using_variance(
                X_selected, threshold=0.01)
            print(
                f"Variance-selected features shape: {X_variance_selected.shape}")

        print("\n" + "="*60)
        print("✅ Feature engineering module ready!")
        print("="*60)
    else:
        print(f"Data file not found: {data_path}")
        print("Please run data_preparation.py first")
