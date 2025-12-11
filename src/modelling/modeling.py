"""
Module for claim severity prediction modeling.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClaimSeverityModeler:
    """
    Builds and evaluates models for claim severity prediction.
    """

    def __init__(self, model_dir: str = None):
        """
        Initialize the claim severity modeler.

        Args:
            model_dir: Directory to save models and results
        """
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path.cwd() / 'models'

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        self.predictions = {}

        # Define model configurations with hyperparameter grids
        self.model_configs = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
            },
            'Lasso': {
                'model': Lasso(random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 100],
                    'max_depth': [-1, 10, 20]
                }
            },
            'DecisionTree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }

        logger.info(f"Initialized ClaimSeverityModeler")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Available models: {list(self.model_configs.keys())}")

    def train_models(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray = None,
                     y_test: np.ndarray = None,
                     use_cv: bool = True,
                     cv_folds: int = 5,
                     save_models: bool = True) -> Dict[str, Any]:
        """
        Train multiple models for claim severity prediction.

        IMPORTANT: y_train and y_test should be LOG-TRANSFORMED values.
        The models are trained on log scale and predictions are converted back.

        Args:
            X_train: Training features
            y_train: Training target (log-transformed: log(1 + TotalClaims))
            X_test: Test features (optional)
            y_test: Test target (optional, log-transformed)
            use_cv: Whether to use cross-validation
            cv_folds: Number of cross-validation folds
            save_models: Whether to save trained models

        Returns:
            Dictionary of trained models
        """
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*60)

        self.results = {}
        cv_results = {}

        # Validate inputs to prevent infinite errors
        self._validate_training_data(X_train, y_train)

        for name, config in self.model_configs.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Training {name}")
            logger.info(f"{'='*40}")

            try:
                # Initialize model
                model = config['model']

                # Validate data before training
                if not self._is_data_valid(X_train, y_train):
                    logger.warning(f"Skipping {name} due to invalid data")
                    continue

                # Cross-validation if requested
                if use_cv:
                    logger.info(
                        f"Performing {cv_folds}-fold cross-validation...")
                    cv = KFold(n_splits=cv_folds,
                               shuffle=True, random_state=42)

                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=cv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )

                    # Convert from log scale to original scale for reporting
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    cv_std = np.sqrt(cv_scores.std())

                    cv_results[name] = {
                        'mean_rmse': cv_rmse,
                        'std_rmse': cv_std,
                        'cv_scores': cv_scores.tolist()
                    }

                    logger.info(f"CV RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")

                # Train the model on full training set
                logger.info("Training model on full training set...")
                model.fit(X_train, y_train)

                # Store the trained model
                self.models[name] = model

                # Save model if requested
                if save_models:
                    self.save_model(model, name)

                # Evaluate on test set if provided
                if X_test is not None and y_test is not None:
                    logger.info("Evaluating on test set...")

                    # Validate test data
                    if not self._is_data_valid(X_test, y_test):
                        logger.warning(
                            f"Skipping evaluation for {name} due to invalid test data")
                        continue

                    # Make predictions on log scale
                    y_pred_log = model.predict(X_test)

                    # Handle any NaN or infinite predictions
                    y_pred_log = self._clean_predictions(y_pred_log)

                    # Convert from log scale to original scale for evaluation
                    y_test_original = np.expm1(y_test)
                    y_pred_original = np.expm1(y_pred_log)

                    # Ensure no negative predictions after transformation
                    y_pred_original = np.maximum(y_pred_original, 0)

                    # Calculate metrics
                    metrics = self._calculate_metrics(
                        y_test_original, y_pred_original)

                    # Store predictions
                    self.predictions[name] = {
                        'y_test_original': y_test_original,
                        'y_pred_original': y_pred_original,
                        'y_test_log': y_test,
                        'y_pred_log': y_pred_log
                    }

                    # Store results
                    self.results[name] = metrics

                    logger.info(f"Test Results:")
                    logger.info(f"  R²: {metrics['r2']:.4f}")
                    logger.info(f"  RMSE: R{metrics['rmse']:,.2f}")
                    logger.info(f"  MAE: R{metrics['mae']:,.2f}")
                    logger.info(f"  MAPE: {metrics['mape']:.2f}%")

                    # Calculate feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = {
                            'importances': model.feature_importances_.tolist(),
                            'sorted_indices': np.argsort(model.feature_importances_)[::-1].tolist()
                        }

                logger.info(f"✅ {name} training complete")

            except Exception as e:
                logger.error(f"❌ Error training {name}: {str(e)}")

        # Identify best model based on R² score
        if self.results:
            # Filter out models with NaN or invalid R² scores
            valid_results = {k: v for k, v in self.results.items()
                             if not np.isnan(v.get('r2', np.nan)) and v.get('r2', -np.inf) > -np.inf}

            if valid_results:
                self.best_model_name = max(valid_results.keys(),
                                           key=lambda x: valid_results[x]['r2'])
                self.best_model = self.models.get(self.best_model_name)

                logger.info(f"\n{'='*60}")
                logger.info(f"🏆 BEST MODEL: {self.best_model_name}")
                logger.info(f"{'='*60}")
                best_metrics = self.results[self.best_model_name]
                logger.info(f"R² Score: {best_metrics['r2']:.4f}")
                logger.info(f"RMSE: R{best_metrics['rmse']:,.2f}")
                logger.info(f"MAE: R{best_metrics['mae']:,.2f}")
                logger.info(f"MAPE: {best_metrics['mape']:.2f}%")
            else:
                logger.warning("No valid models with R² scores found")

        # Save cross-validation results if available
        if cv_results:
            cv_file = self.model_dir / 'cross_validation_results.json'
            with open(cv_file, 'w') as f:
                json.dump(cv_results, f, indent=2)
            logger.info(f"💾 Saved cross-validation results to: {cv_file}")

        return self.models

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data before modeling."""
        logger.info("Validating training data...")

        # Check for NaN values
        x_nan = np.isnan(X).any()
        y_nan = np.isnan(y).any()

        if x_nan or y_nan:
            logger.warning(f"NaN values found: X={x_nan}, y={y_nan}")

        # Check for infinite values
        x_inf = np.isinf(X).any()
        y_inf = np.isinf(y).any()

        if x_inf or y_inf:
            logger.warning(f"Infinite values found: X={x_inf}, y={y_inf}")

        # Check data shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")

        logger.info(
            f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(
            f"Target range (log scale): [{y.min():.2f}, {y.max():.2f}]")

    def _is_data_valid(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Check if data is valid for training/prediction."""
        if X is None or y is None:
            return False

        if len(X) == 0 or len(y) == 0:
            return False

        if np.isnan(X).any() or np.isnan(y).any():
            logger.warning("NaN values detected in data")
            return False

        if np.isinf(X).any() or np.isinf(y).any():
            logger.warning("Infinite values detected in data")
            return False

        return True

    def _clean_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """Clean predictions by handling NaN and infinite values."""
        y_pred_clean = y_pred.copy()

        # Replace NaN with median
        if np.isnan(y_pred_clean).any():
            median_val = np.nanmedian(y_pred_clean)
            y_pred_clean = np.nan_to_num(y_pred_clean, nan=median_val)

        # Replace infinite values with finite bounds
        if np.isinf(y_pred_clean).any():
            finite_vals = y_pred_clean[np.isfinite(y_pred_clean)]
            if len(finite_vals) > 0:
                max_val = np.max(finite_vals)
                min_val = np.min(finite_vals)
                y_pred_clean = np.clip(y_pred_clean, min_val, max_val)

        return y_pred_clean

    def hyperparameter_tuning(self,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              model_name: str = None,
                              cv_folds: int = 3,
                              n_iter: int = 20) -> Dict:
        """
        Perform hyperparameter tuning for one or all models.

        Args:
            X_train: Training features
            y_train: Training target (log-transformed)
            model_name: Name of model to tune (None for all models)
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for random search

        Returns:
            Dictionary of tuned models
        """
        if model_name and model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not in configuration")

        models_to_tune = [model_name] if model_name else list(
            self.model_configs.keys())

        tuned_models = {}

        for name in models_to_tune:
            logger.info(f"\n{'='*40}")
            logger.info(f"HYPERPARAMETER TUNING: {name}")
            logger.info(f"{'='*40}")

            config = self.model_configs[name]

            try:
                # Validate data before tuning
                if not self._is_data_valid(X_train, y_train):
                    logger.warning(
                        f"Skipping tuning for {name} due to invalid data")
                    continue

                # Use randomized search for faster tuning
                random_search = RandomizedSearchCV(
                    estimator=config['model'],
                    param_distributions=config['params'],
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )

                logger.info(
                    f"Performing randomized search with {n_iter} iterations...")
                random_search.fit(X_train, y_train)

                logger.info(f"Best parameters: {random_search.best_params_}")
                best_cv_score = np.sqrt(-random_search.best_score_)
                logger.info(
                    f"Best CV score (RMSE on log scale): {best_cv_score:.4f}")

                # Update model with best parameters
                best_model = random_search.best_estimator_
                self.models[name] = best_model
                tuned_models[name] = best_model

                # Save best parameters
                params_file = self.model_dir / f"{name}_best_params.json"
                with open(params_file, 'w') as f:
                    json.dump(random_search.best_params_, f, indent=2)

                logger.info(f"💾 Saved best parameters to: {params_file}")

                # Save tuned model
                self.save_model(best_model, f"{name}_tuned")

            except Exception as e:
                logger.error(f"Error tuning {name}: {str(e)}")

        return tuned_models

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics with robust handling."""
        # Handle any NaN or infinite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
            logger.warning("No valid data points for metric calculation")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'mpe': np.nan,
                'smape': np.nan
            }

        # Check for zero or negative values that could cause issues
        if (y_true_clean <= 0).any():
            logger.warning(
                "Zero or negative values in y_true, using adjusted MAPE calculation")

        # Calculate metrics with error handling
        try:
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            mae = mean_absolute_error(y_true_clean, y_pred_clean)

            # Calculate R² with fallback
            try:
                r2 = r2_score(y_true_clean, y_pred_clean)
            except:
                r2 = 1 - (np.sum((y_true_clean - y_pred_clean) ** 2) /
                          np.sum((y_true_clean - np.mean(y_true_clean)) ** 2))

            # Calculate MAPE with protection against division by zero
            nonzero_mask = y_true_clean != 0
            if nonzero_mask.any():
                mape = np.mean(np.abs((y_true_clean[nonzero_mask] - y_pred_clean[nonzero_mask]) /
                                      y_true_clean[nonzero_mask])) * 100
            else:
                mape = np.nan

            # Calculate MPE
            if nonzero_mask.any():
                mpe = np.mean((y_true_clean[nonzero_mask] - y_pred_clean[nonzero_mask]) /
                              y_true_clean[nonzero_mask]) * 100
            else:
                mpe = np.nan

            # Calculate SMAPE
            denominator = np.abs(y_true_clean) + np.abs(y_pred_clean)
            nonzero_denom = denominator != 0
            if nonzero_denom.any():
                smape = 100 * np.mean(2 * np.abs(y_pred_clean[nonzero_denom] - y_true_clean[nonzero_denom]) /
                                      denominator[nonzero_denom])
            else:
                smape = np.nan

            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'mpe': mpe,
                'smape': smape
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics = {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'mpe': np.nan,
                'smape': np.nan
            }

        return metrics

    def evaluate_model(self,
                       model: Any,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       model_name: str = "Model") -> Dict:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target (log-transformed)
            model_name: Name for the model

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")

        # Validate data
        if not self._is_data_valid(X_test, y_test):
            logger.warning(f"Cannot evaluate {model_name} - invalid test data")
            return None

        # Make predictions on log scale
        y_pred_log = model.predict(X_test)

        # Clean predictions
        y_pred_log = self._clean_predictions(y_pred_log)

        # Convert from log scale to original scale
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred_log)

        # Ensure no negative predictions
        y_pred_original = np.maximum(y_pred_original, 0)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test_original, y_pred_original)

        # Create evaluation report
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': {
                'actual': y_test_original.tolist(),
                'predicted': y_pred_original.tolist(),
                'log_actual': y_test.tolist(),
                'log_predicted': y_pred_log.tolist()
            },
            'error_analysis': {
                'mean_error': float(np.mean(y_test_original - y_pred_original)),
                'median_error': float(np.median(y_test_original - y_pred_original)),
                'std_error': float(np.std(y_test_original - y_pred_original)),
                'max_underprediction': float(np.max(y_test_original - y_pred_original)),
                'max_overprediction': float(np.max(y_pred_original - y_test_original))
            }
        }

        # Generate visualizations
        try:
            self._generate_evaluation_plots(
                y_test_original, y_pred_original, model_name)
        except Exception as e:
            logger.warning(f"Could not generate evaluation plots: {str(e)}")

        return report

    def _generate_evaluation_plots(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   model_name: str) -> None:
        """Generate evaluation plots."""
        plot_dir = self.model_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 1. Actual vs Predicted scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=10, color='steelblue')
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([0, max_val], [0, max_val], 'r--',
                     lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Claims (R)', fontsize=12)
        axes[0].set_ylabel('Predicted Claims (R)', fontsize=12)
        axes[0].set_title(
            f'Actual vs Predicted - {model_name}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=10, color='coral')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Claims (R)', fontsize=12)
        axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        axes[1].set_title(
            f'Residual Plot - {model_name}', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 3. Distribution of errors
        axes[2].hist(residuals, bins=50, edgecolor='black',
                     alpha=0.7, color='seagreen')
        axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[2].axvline(x=np.mean(residuals), color='blue', linestyle='-', lw=2,
                        label=f'Mean: R{np.mean(residuals):,.0f}')
        axes[2].set_xlabel('Prediction Error (R)', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title(
            f'Error Distribution - {model_name}', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 4. Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[3])
        axes[3].set_title(
            f'Q-Q Plot of Residuals - {model_name}', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)

        # 5. Prediction error vs actual
        abs_errors = np.abs(residuals)
        axes[4].scatter(y_true, abs_errors, alpha=0.5, s=10, color='purple')
        axes[4].set_xlabel('Actual Claims (R)', fontsize=12)
        axes[4].set_ylabel('Absolute Error (R)', fontsize=12)
        axes[4].set_title(
            f'Absolute Error vs Actual - {model_name}', fontsize=14, fontweight='bold')
        axes[4].grid(True, alpha=0.3)

        # 6. Cumulative distribution of absolute percentage error
        ape = np.abs(residuals / (y_true + 1e-10)) * 100
        sorted_ape = np.sort(ape)
        cdf = np.arange(1, len(sorted_ape) + 1) / len(sorted_ape)
        axes[5].plot(sorted_ape, cdf, 'b-', lw=2)
        axes[5].set_xlabel('Absolute Percentage Error (%)', fontsize=12)
        axes[5].set_ylabel('Cumulative Probability', fontsize=12)
        axes[5].set_title(f'APE CDF - {model_name}',
                          fontsize=14, fontweight='bold')
        axes[5].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = plot_dir / f'{model_name}_evaluation.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Saved evaluation plots to: {plot_file}")

    def generate_model_comparison_report(self,
                                         feature_names: List[str] = None) -> pd.DataFrame:
        """
        Generate comparison report of all models.

        Args:
            feature_names: List of feature names for importance analysis

        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")

        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE': f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else 'N/A',
                'SMAPE': f"{metrics['smape']:.2f}%" if not np.isnan(metrics['smape']) else 'N/A',
                'MPE': f"{metrics['mpe']:.2f}%" if not np.isnan(metrics['mpe']) else 'N/A'
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by R², handling NaN values
        comparison_df['R²_numeric'] = pd.to_numeric(
            comparison_df['R²'], errors='coerce')
        comparison_df = comparison_df.sort_values(
            'R²_numeric', ascending=False)
        comparison_df = comparison_df.drop(columns=['R²_numeric'])

        # Save to CSV
        report_file = self.model_dir / 'model_comparison.csv'
        comparison_df.to_csv(report_file, index=False)

        # Save detailed results to JSON
        detailed_results = {
            'model_comparison': comparison_df.to_dict('records'),
            'detailed_metrics': self.results,
            'best_model': {
                'name': self.best_model_name,
                'metrics': self.results.get(self.best_model_name, {})
            }
        }

        if feature_names and self.best_model_name in self.feature_importance:
            # Add feature importance for best model
            importances = self.feature_importance[self.best_model_name]['importances']
            sorted_idx = self.feature_importance[self.best_model_name]['sorted_indices']

            top_features = []
            for idx in sorted_idx[:20]:  # Top 20 features
                if idx < len(feature_names):
                    top_features.append({
                        'feature': feature_names[idx],
                        'importance': importances[idx]
                    })

            detailed_results['feature_importance'] = {
                'best_model': self.best_model_name,
                'top_features': top_features
            }

        json_file = self.model_dir / 'model_comparison.json'
        with open(json_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        logger.info(f"📊 Model comparison saved to:")
        logger.info(f"  CSV: {report_file}")
        logger.info(f"  JSON: {json_file}")

        # Print comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def plot_model_comparison(self, metric: str = 'r2'):
        """
        Create visualization comparing model performance.

        Args:
            metric: Metric to compare ('r2', 'rmse', 'mae', 'mape')
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")

        # Filter out models with NaN values for the selected metric
        models_with_metrics = []
        scores = []

        for model_name, metrics in self.results.items():
            if metric in metrics and not np.isnan(metrics[metric]):
                models_with_metrics.append(model_name)
                scores.append(metrics[metric])

        if not models_with_metrics:
            logger.warning(f"No models with valid {metric} scores found")
            return

        # Determine ylabel and color based on metric
        metric_config = {
            'r2': ('R² Score', 'skyblue'),
            'rmse': ('RMSE (R)', 'lightcoral'),
            'mae': ('MAE (R)', 'lightgreen'),
            'mape': ('MAPE (%)', 'gold')
        }

        if metric not in metric_config:
            raise ValueError(f"Unknown metric: {metric}")

        ylabel, color = metric_config[metric]

        # Create figure
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models_with_metrics, scores,
                       color=color, edgecolor='black')

        # Highlight best model
        if self.best_model_name in models_with_metrics:
            best_idx = models_with_metrics.index(self.best_model_name)
            bars[best_idx].set_color('darkorange')
            bars[best_idx].set_edgecolor('darkred')

        plt.xlabel('Models', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'Model Comparison: {ylabel}',
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if metric in ['r2']:
                label = f'{height:.3f}'
            elif metric in ['rmse', 'mae']:
                if height > 1e6:
                    label = f'{height:.2e}'
                else:
                    label = f'{height:,.0f}'
            else:  # mape
                label = f'{height:.1f}%'

            plt.text(bar.get_x() + bar.get_width()/2., height,
                     label, ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plot_file = self.model_dir / f'model_comparison_{metric}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Saved model comparison plot to: {plot_file}")

        # Create combined comparison plot
        self._create_combined_comparison_plot()

    def _create_combined_comparison_plot(self):
        """Create combined comparison plot with multiple metrics."""
        if not self.results:
            return

        models = list(self.results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics_config = [
            ('r2', 'R² Score', 'skyblue'),
            ('rmse', 'RMSE (R)', 'lightcoral'),
            ('mae', 'MAE (R)', 'lightgreen'),
            ('mape', 'MAPE (%)', 'gold')
        ]

        for idx, (metric, title, color) in enumerate(metrics_config):
            # Collect scores for each model
            scores = []
            valid_models = []

            for model in models:
                if metric in self.results[model] and not np.isnan(self.results[model][metric]):
                    scores.append(self.results[model][metric])
                    valid_models.append(model)

            if not scores:
                axes[idx].text(0.5, 0.5, f'No valid {metric} data',
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(title, fontsize=12, fontweight='bold')
                continue

            bars = axes[idx].bar(valid_models, scores,
                                 color=color, edgecolor='black')

            # Highlight best model (only in first plot for clarity)
            if self.best_model_name in valid_models and idx == 0:
                best_idx = valid_models.index(self.best_model_name)
                bars[best_idx].set_color('darkorange')
                bars[best_idx].set_edgecolor('darkred')

            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].set_xticklabels(
                valid_models, rotation=45, ha='right', fontsize=9)
            axes[idx].grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if metric == 'r2':
                    label = f'{height:.3f}'
                elif metric in ['rmse', 'mae']:
                    if height > 1e6:
                        label = f'{height:.2e}'
                    else:
                        label = f'{height:,.0f}'
                else:  # mape
                    label = f'{height:.1f}%'

                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                               label, ha='center', va='bottom', fontsize=8)

        plt.suptitle('Model Performance Comparison',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_file = self.model_dir / 'model_comparison_combined.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Saved combined comparison plot to: {plot_file}")

    def save_model(self, model: Any, model_name: str) -> None:
        """Save model to file."""
        model_file = self.model_dir / f'{model_name}.pkl'

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"💾 Saved model to: {model_file}")

    def load_model(self, model_name: str) -> Any:
        """Load model from file."""
        model_file = self.model_dir / f'{model_name}.pkl'

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"📂 Loaded model from: {model_file}")
        return model

    def predict(self,
                model: Any,
                X: np.ndarray,
                return_log: bool = False) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model: Trained model
            X: Input features
            return_log: Whether to return log-transformed predictions

        Returns:
            Predictions (original scale or log scale)
        """
        # Validate input
        if not self._is_data_valid(X, np.zeros(X.shape[0])):
            logger.warning("Invalid input data for prediction")
            return None

        # Make predictions on log scale
        y_pred_log = model.predict(X)

        # Clean predictions
        y_pred_log = self._clean_predictions(y_pred_log)

        if return_log:
            return y_pred_log
        else:
            # Convert to original scale
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
            return y_pred

    def get_model_summary(self) -> Dict:
        """Get summary of all trained models."""
        summary = {
            'best_model': self.best_model_name,
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'results_available': bool(self.results),
            'predictions_available': bool(self.predictions),
            'feature_importance_available': bool(self.feature_importance)
        }

        if self.best_model_name and self.best_model_name in self.results:
            summary['best_model_metrics'] = self.results[self.best_model_name]

        return summary

    def generate_feature_importance_plot(self,
                                         feature_names: List[str],
                                         top_n: int = 20) -> None:
        """
        Generate feature importance plot for best model.

        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
        """
        if not self.best_model_name or self.best_model_name not in self.feature_importance:
            logger.warning(
                "No feature importance data available for best model")
            return

        if not feature_names:
            logger.warning("Feature names not provided")
            return

        importances = self.feature_importance[self.best_model_name]['importances']
        sorted_idx = self.feature_importance[self.best_model_name]['sorted_indices']

        # Take top N features
        top_idx = sorted_idx[:min(top_n, len(feature_names))]
        top_features = [feature_names[i] for i in top_idx]
        top_importances = [importances[i] for i in top_idx]

        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(top_features))

        plt.barh(y_pos, top_importances, color='steelblue', edgecolor='black')
        plt.yticks(y_pos, top_features)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {len(top_features)} Feature Importances - {self.best_model_name}',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(top_importances):
            plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        plot_file = self.model_dir / \
            f'feature_importance_{self.best_model_name}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Saved feature importance plot to: {plot_file}")

        # Save feature importance to CSV
        importance_df = pd.DataFrame({
            'feature': top_features,
            'importance': top_importances
        })
        csv_file = self.model_dir / \
            f'feature_importance_{self.best_model_name}.csv'
        importance_df.to_csv(csv_file, index=False)
        logger.info(f"💾 Saved feature importance data to: {csv_file}")


# Main execution block for testing
if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    print("="*80)
    print("CLAIM SEVERITY MODELING MODULE")
    print("="*80)

    # Create a test instance
    modeler = ClaimSeverityModeler()

    print("\nThis module provides:")
    print("1. Multiple regression model implementations")
    print("2. Hyperparameter tuning with cross-validation")
    print("3. Model evaluation with comprehensive metrics")
    print("4. Visualization of results")
    print("5. Model comparison and selection")
    print("6. Feature importance analysis")

    print("\nAvailable models:")
    for model_name in modeler.model_configs.keys():
        print(f"  • {model_name}")

    print("\n" + "="*80)
    print("✅ Modeling module ready for integration with pipeline!")
    print("="*80)
