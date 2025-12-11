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
from scipy.stats import spearmanr
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
        self.training_stats_ = {}  # Store training statistics for test data

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
                               X_test: pd.DataFrame = None,
                               y_train: pd.Series = None) -> Tuple:
        """
        Fit preprocessing on training data and transform both train and test.
        Ensures test data has same columns as training data.

        Args:
            X_train: Training features
            X_test: Test features (optional)
            y_train: Training target (optional, for feature scaling reference only)

        Returns:
            Tuple of (X_train_transformed, X_test_transformed or None)
        """
        logger.info("Fitting and transforming features...")

        # Create features BEFORE preprocessing (without target leakage)
        X_train_features = self.create_additional_features(
            X_train, is_training=True)

        # Store training columns
        self.training_columns_ = X_train_features.columns.tolist()
        logger.info(f"Training data has {len(self.training_columns_)} columns")

        # Identify feature types
        self.identify_feature_types(X_train_features)

        # Create preprocessing pipeline
        self.create_preprocessing_pipeline()

        # Fit and transform training data
        logger.info("Fitting preprocessor on training data...")
        X_train_transformed = self.preprocessor.fit_transform(X_train_features)

        # Get feature names
        self._get_feature_names(X_train_features)

        # Transform test data if provided
        X_test_transformed = None
        if X_test is not None:
            logger.info("Transforming test data...")

            # Create features for test data WITHOUT using any target information
            X_test_features = self.create_additional_features(
                X_test, is_training=False)

            # Align test data to match training columns
            X_test_aligned = self._align_test_data(X_test_features)

            # Transform
            try:
                X_test_transformed = self.preprocessor.transform(
                    X_test_aligned)
                logger.info(
                    f"  Test features shape: {X_test_transformed.shape}")
            except Exception as e:
                logger.error(f"Error transforming test data: {str(e)}")
                logger.error(f"Training columns: {self.training_columns_}")
                logger.error(
                    f"Test columns: {X_test_features.columns.tolist()}")
                raise

        logger.info(f"  Training features shape: {X_train_transformed.shape}")
        logger.info(
            f"  Total features after preprocessing: {X_train_transformed.shape[1]}")

        return X_train_transformed, X_test_transformed

    def _align_test_data(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Align test data columns to match training data.

        Args:
            X_test: Test data to align

        Returns:
            Aligned test data
        """
        X_aligned = X_test.copy()

        # Add missing columns with appropriate defaults
        missing_cols = set(self.training_columns_) - set(X_aligned.columns)
        for col in missing_cols:
            if col in self.numerical_features:
                X_aligned[col] = 0
            elif col in self.categorical_features:
                X_aligned[col] = 'missing'
            else:
                X_aligned[col] = 0

        # Remove extra columns
        extra_cols = set(X_aligned.columns) - set(self.training_columns_)
        if extra_cols:
            X_aligned = X_aligned.drop(columns=list(extra_cols))

        # Ensure correct order
        X_aligned = X_aligned[self.training_columns_]

        return X_aligned

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

        # Create features for new data
        X_new_features = self.create_additional_features(
            X_new, is_training=False)

        # Align columns to match training
        X_new_aligned = self._align_columns(X_new_features)

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

    def create_additional_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create additional interaction and polynomial features WITHOUT target leakage.

        Args:
            X: Input features
            is_training: Whether this is training data (store statistics if True)

        Returns:
            DataFrame with additional features
        """
        logger.info("Creating additional advanced features...")

        X_enhanced = X.copy()
        original_cols = X_enhanced.columns.tolist()

        # Ensure we have numerical columns
        if self.numerical_features is None:
            self.identify_feature_types(X)

        # 1. Polynomial features for important numerical columns (SAFE - no target leakage)
        important_num_cols = ['VehicleAge', 'SumInsured',
                              'TotalPremium', 'cubiccapacity']
        available_num_cols = [
            col for col in important_num_cols if col in X_enhanced.columns]

        for col in available_num_cols:
            try:
                # Check if column has enough variation
                if X_enhanced[col].nunique() > 1:
                    # Store min/max for clipping if training
                    if is_training:
                        self.training_stats_[
                            f'{col}_min'] = X_enhanced[col].min()
                        self.training_stats_[
                            f'{col}_max'] = X_enhanced[col].max()

                    # Square (clip to prevent overflow)
                    col_squared = X_enhanced[col] ** 2
                    X_enhanced[f'{col}_squared'] = np.clip(
                        col_squared, -1e10, 1e10)

                    # Cube (clip to prevent overflow)
                    col_cubed = X_enhanced[col] ** 3
                    X_enhanced[f'{col}_cubed'] = np.clip(
                        col_cubed, -1e10, 1e10)

                    # Log transform (add small epsilon to avoid log(0))
                    eps = 1e-10
                    X_enhanced[f'log_{col}'] = np.log1p(
                        np.abs(X_enhanced[col] + eps))
            except Exception as e:
                logger.warning(
                    f"Could not create polynomial features for {col}: {str(e)}")

        # 2. Interaction features (SAFE - no target leakage)
        # Age-Value interaction (but be careful with scale)
        try:
            if 'VehicleAge' in X_enhanced.columns and 'SumInsured' in X_enhanced.columns:
                # Clip values to prevent overflow
                age_clipped = np.clip(X_enhanced['VehicleAge'], 0, 50)
                sum_insured_clipped = np.clip(
                    X_enhanced['SumInsured'], 1000, 1000000)

                X_enhanced['Age_Value_Interaction'] = age_clipped * \
                    sum_insured_clipped

                # Ratio with protection against division by zero
                denominator = sum_insured_clipped.replace(0, np.nan)
                X_enhanced['Age_Value_Ratio'] = age_clipped / denominator
                X_enhanced['Age_Value_Ratio'] = X_enhanced['Age_Value_Ratio'].fillna(
                    0)

                # Clip extreme values
                X_enhanced['Age_Value_Ratio'] = np.clip(
                    X_enhanced['Age_Value_Ratio'], -100, 100)
        except Exception as e:
            logger.warning(f"Could not create interaction features: {str(e)}")

        # 3. Premium ratio (SAFE - no target leakage)
        try:
            if 'TotalPremium' in X_enhanced.columns and 'SumInsured' in X_enhanced.columns:
                # Clip to prevent extreme values
                premium_clipped = np.clip(
                    X_enhanced['TotalPremium'], 100, 10000)
                sum_insured_clipped = np.clip(
                    X_enhanced['SumInsured'], 1000, 1000000)

                denominator = sum_insured_clipped.replace(0, np.nan)
                X_enhanced['Premium_Per_Value'] = premium_clipped / denominator
                X_enhanced['Premium_Per_Value'] = X_enhanced['Premium_Per_Value'].fillna(
                    0)

                # Clip to reasonable range
                X_enhanced['Premium_Per_Value'] = np.clip(
                    X_enhanced['Premium_Per_Value'], 0, 0.1)
        except Exception as e:
            logger.warning(f"Could not create premium ratio: {str(e)}")

        # 4. Risk score features (using existing features only, no target)
        risk_factors = []
        if 'VehicleAge' in X_enhanced.columns:
            # Old vehicles (based on VehicleAge only)
            risk_factors.append((X_enhanced['VehicleAge'] > 10).astype(int))

        if 'cubiccapacity' in X_enhanced.columns:
            # Large engine (use median from training if available, otherwise calculate)
            if is_training:
                median_cc = X_enhanced['cubiccapacity'].median()
                self.training_stats_['cubiccapacity_median'] = median_cc
            else:
                median_cc = self.training_stats_.get(
                    'cubiccapacity_median', X_enhanced['cubiccapacity'].median())

            risk_factors.append(
                (X_enhanced['cubiccapacity'] > median_cc).astype(int))

        if 'HasHighPower' in X_enhanced.columns:
            risk_factors.append(X_enhanced['HasHighPower'])

        if risk_factors:
            X_enhanced['Risk_Score'] = sum(risk_factors) / len(risk_factors)

        # 5. Binning numerical features (store bin edges if training)
        try:
            if 'VehicleAge' in X_enhanced.columns:
                bins = [0, 3, 7, 12, 20, 100]
                labels = ['New', 'Young', 'Middle', 'Old', 'VeryOld']

                X_enhanced['VehicleAge'] = X_enhanced['VehicleAge'].clip(
                    0, 100)
                X_enhanced['VehicleAge_Binned'] = pd.cut(
                    X_enhanced['VehicleAge'],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )

                if is_training:
                    self.training_stats_['VehicleAge_bins'] = bins
        except Exception as e:
            logger.warning(f"Could not bin VehicleAge: {str(e)}")

        # 6. Binning SumInsured (handle edge cases)
        try:
            if 'SumInsured' in X_enhanced.columns:
                if is_training:
                    # On training data, calculate percentiles
                    try:
                        # Try qcut first
                        X_enhanced['SumInsured_Binned'] = pd.qcut(
                            X_enhanced['SumInsured'],
                            q=5,
                            labels=['VeryLow', 'Low',
                                    'Medium', 'High', 'VeryHigh'],
                            duplicates='drop'
                        )
                        # Store the bins
                        if hasattr(X_enhanced['SumInsured_Binned'].cat, 'categories'):
                            self.training_stats_['SumInsured_bins'] = 'qcut_5'
                    except:
                        # If qcut fails, use manual bins
                        percentiles = X_enhanced['SumInsured'].quantile(
                            [0, 0.2, 0.4, 0.6, 0.8, 1.0])
                        X_enhanced['SumInsured_Binned'] = pd.cut(
                            X_enhanced['SumInsured'],
                            bins=percentiles.unique(),
                            labels=['VeryLow', 'Low',
                                    'Medium', 'High', 'VeryHigh'],
                            include_lowest=True
                        )
                        self.training_stats_[
                            'SumInsured_bins'] = percentiles.tolist()
                else:
                    # On test data, use default bins or skip
                    X_enhanced['SumInsured_Binned'] = 'Medium'  # Default value
        except Exception as e:
            logger.warning(f"Could not bin SumInsured: {str(e)}")

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
                                          threshold: float = 0.05) -> pd.DataFrame:
        """
        Select features based on correlation with target.
        But only for feature selection, not for creating new features.

        Args:
            X: Feature DataFrame
            y: Target variable
            threshold: Absolute correlation threshold

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting features with |correlation| > {threshold}...")

        # Select only numerical features
        X_numeric = X.select_dtypes(include=[np.number]).copy()

        if len(X_numeric.columns) == 0:
            logger.warning("No numerical features for correlation selection")
            return X

        # Filter out features with extreme values that might cause correlation issues
        safe_features = []
        for col in X_numeric.columns:
            # Check for finite values
            if X_numeric[col].notna().all():
                # Check for reasonable range (no infinities)
                if not np.any(np.isinf(X_numeric[col])):
                    if X_numeric[col].nunique() > 1:
                        safe_features.append(col)

        if not safe_features:
            return X

        X_safe = X_numeric[safe_features]

        # Calculate correlations
        correlations = []
        for col in X_safe.columns:
            try:
                # Use rank correlation to be robust to outliers
                corr, _ = spearmanr(X_safe[col], y)
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
            except:
                continue

        if not correlations:
            logger.warning("Could not calculate correlations")
            return X

        # Sort by correlation
        correlations.sort(key=lambda x: x[1], reverse=True)

        # Select features above threshold
        selected_numeric = [feat for feat,
                            corr in correlations if corr > threshold]

        if not selected_numeric:
            logger.warning(
                f"No features above threshold {threshold}. Using all numeric features.")
            selected_numeric = safe_features

        # Include categorical features
        selected_features = selected_numeric.copy()
        if self.categorical_features:
            selected_features.extend(self.categorical_features)

        logger.info(f"Selected {len(selected_features)} features")

        # Log top correlations
        if correlations:
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

        # Save both preprocessor and training stats
        save_data = {
            'preprocessor': self.preprocessor,
            'training_stats': self.training_stats_,
            'training_columns': self.training_columns_,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"💾 Saved preprocessor to: {filepath}")

    def load_preprocessor(self, filepath: str) -> None:
        """Load preprocessor from file."""
        import pickle

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.preprocessor = save_data['preprocessor']
        self.training_stats_ = save_data.get('training_stats', {})
        self.training_columns_ = save_data.get('training_columns', [])
        self.numerical_features = save_data.get('numerical_features', [])
        self.categorical_features = save_data.get('categorical_features', [])

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

        # Create additional features WITHOUT target
        X_enhanced = feature_engineer.create_additional_features(
            X, is_training=True)
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
