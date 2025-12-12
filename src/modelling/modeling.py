"""
Modeling pipeline for AlphaCare Insurance Solutions (ACIS)

This script expects the prepared dataset produced by your `ClaimDataPreparer`:
- ./data/processed/claim_data_prepared.csv

Features:
- Builds two modeling tasks:
  1. Claim severity regression (on policies with TotalClaims > 0)
  2. Claim probability classification (binary: HasClaim / HighClaim)
- Trains baseline models: Linear Regression, RandomForest, and optionally XGBoost / CatBoost
- Evaluates models (RMSE, R2 for regression; AUC/accuracy for classification)
- Optional SHAP explainability for the best model
- Saves trained models and preprocessing artifacts to ./models/

Usage (example):
    python modeling_pipeline.py --data data/processed/claim_data_prepared.csv --out models/

Make sure required packages are installed: scikit-learn, pandas, numpy, joblib. 
For XGBoost/CatBoost/SHAP install optionally.
"""

import argparse
import logging
from pathlib import Path
import joblib
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


# Optional libs
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier, CatBoostError
    HAS_CAT = True

except ImportError:
    HAS_CAT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

FORCE_NO_CATBOOST = True  # Set to False to enable CatBoost
# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s - %(message)s')
logger = logging.getLogger('modeling')

RANDOM_STATE = 42


# -----------------------
# Helpers
# -----------------------

def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    logger.info("Loading prepared data from: %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows and %d columns", len(df), df.shape[1])
    return df


def select_features(df: pd.DataFrame, drop_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X, y for regression (TotalClaims) or classification (HasClaim/HighClaim) as required.

    This helper does minimal additional cleaning: drops exact identifier columns and columns that leak the target.
    """
    if drop_cols is None:
        drop_cols = ['PolicyID']

    df = df.copy()
    # Remove obvious leakage or identifiers
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df


def build_preprocessor(df: pd.DataFrame, categorical_limit_onehot: int = 10):
    """
    Build preprocessing pipeline with proper imputation.
    - Numeric: SimpleImputer(median) → StandardScaler
    - Low-card categorical: SimpleImputer(mode) → OneHotEncoder
    - High-card categorical: SimpleImputer(mode) → OrdinalEncoder
    """

    # Identify columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols
        if c not in ['TotalClaims']
        and not c.lower().startswith('log_totalclaims')
    ]

    cat_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    low_card = [c for c in cat_cols if df[c].nunique() <=
                categorical_limit_onehot]
    high_card = [c for c in cat_cols if df[c].nunique() >
                 categorical_limit_onehot]

    logger.info("Numeric cols: %d, low-cardinal cats: %d, high-card cats: %d",
                len(numeric_cols), len(low_card), len(high_card))

    transformers = []

    # --------------------------
    # NUMERIC PIPELINE
    # --------------------------
    if numeric_cols:
        num_pipe = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="median")),
            ('scale', StandardScaler())
        ])
        transformers.append(('num', num_pipe, numeric_cols))

    # --------------------------
    # LOW-CARDINALITY CATEGORICAL
    # --------------------------
    if low_card:
        ohe_pipe = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="most_frequent")),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(("ohe", ohe_pipe, low_card))

    # --------------------------
    # HIGH-CARDINALITY CATEGORICAL
    # --------------------------
    if high_card:
        ord_pipe = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="most_frequent")),
            ('ord', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])
        transformers.append(("ord", ord_pipe, high_card))

    # Build final column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor, numeric_cols, low_card + high_card


# -----------------------
# Modeling tasks
# -----------------------

def train_regression_models(X_train, y_train, X_val, y_val, preprocessor, model_dir: Path) -> Dict:
    """Train a few regression models and return evaluation results and best models."""
    results = {}
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) Linear Regression baseline
    pipe_lr = Pipeline([('pre', preprocessor), ('lr', LinearRegression())])
    logger.info('Training Linear Regression...')
    pipe_lr.fit(X_train, y_train)
    preds = pipe_lr.predict(X_val)
    results['linear'] = {
        # 'rmse': mean_squared_error(y_val, preds, squared=False),
        'rmse': np.sqrt(mean_squared_error(y_val, preds)),
        'r2': r2_score(y_val, preds),
        'model': pipe_lr
    }
    joblib.dump(pipe_lr, model_dir / 'reg_linear.pkl')

    # 2) Random Forest
    pipe_rf = Pipeline([('pre', preprocessor), ('rf', RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))])
    logger.info('Training RandomForestRegressor...')
    pipe_rf.fit(X_train, y_train)
    preds = pipe_rf.predict(X_val)
    results['rf'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, preds)),
        'r2': r2_score(y_val, preds),
        'model': pipe_rf
    }
    joblib.dump(pipe_rf, model_dir / 'reg_rf.pkl')

    # 3) XGBoost (optional)
    if HAS_XGB:
        logger.info('Training XGBoostRegressor...')
        xgb_reg = xgb.XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        pipe_xgb = Pipeline([('pre', preprocessor), ('xgb', xgb_reg)])
        pipe_xgb.fit(X_train, y_train)
        preds = pipe_xgb.predict(X_val)
        results['xgb'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, preds)),
            'r2': r2_score(y_val, preds),
            'model': pipe_xgb
        }
        joblib.dump(pipe_xgb, model_dir / 'reg_xgb.pkl')

    # 4) CatBoost (optional) - can accept categoricals directly, so we train a CatBoost model separately
    if HAS_CAT and not FORCE_NO_CATBOOST:
        try:
            cat_cols = X_train.select_dtypes(
                include=['object', 'category']).columns.tolist()
            logger.info(
                'Training CatBoostRegressor (handles categoricals natively)...')
            cbr = CatBoostRegressor(
                iterations=500, learning_rate=0.05, random_seed=RANDOM_STATE, verbose=0)
            # CatBoost requires raw arrays; we pass df and specify cat_features indices
            cbr.fit(X_train, y_train, cat_features=cat_cols)
            preds = cbr.predict(X_val)
            results['catboost'] = {
                'rmse': np.sqrt(mean_squared_error(y_val, preds)),
                'r2': r2_score(y_val, preds),
                'model': cbr
            }
            joblib.dump(cbr, model_dir / 'reg_catboost.pkl')
        except (CatBoostError, ValueError, TypeError) as e:
            logger.warning(f"CatBoost train failed: {e}")

    # Log results
    for name, res in results.items():
        logger.info(
            f"Model {name}: RMSE={res['rmse']:.4f}, R2={res['r2']:.4f}")

    return results


def train_classification_models(X_train, y_train, X_val, y_val, preprocessor, model_dir: Path) -> Dict:
    results = {}
    model_dir.mkdir(parents=True, exist_ok=True)

    # Logistic baseline
    pipe_log = Pipeline(
        [('pre', preprocessor), ('log', LogisticRegression(max_iter=2000))])
    logger.info('Training LogisticRegression...')
    pipe_log.fit(X_train, y_train)
    preds_proba = pipe_log.predict_proba(X_val)[:, 1]
    preds = pipe_log.predict(X_val)
    results['logistic'] = {
        'auc': roc_auc_score(y_val, preds_proba),
        'accuracy': accuracy_score(y_val, preds),
        'model': pipe_log
    }
    joblib.dump(pipe_log, model_dir / 'clf_logistic.pkl')

    # RandomForestClassifier
    pipe_rf = Pipeline([('pre', preprocessor), ('rf', RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))])
    logger.info('Training RandomForestClassifier...')
    pipe_rf.fit(X_train, y_train)
    preds_proba = pipe_rf.predict_proba(X_val)[:, 1]
    preds = pipe_rf.predict(X_val)
    results['rf'] = {
        'auc': roc_auc_score(y_val, preds_proba),
        'accuracy': accuracy_score(y_val, preds),
        'model': pipe_rf
    }
    joblib.dump(pipe_rf, model_dir / 'clf_rf.pkl')

    if HAS_XGB:
        logger.info('Training XGBoostClassifier...')
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE, eval_metric='auc')
        pipe_xgb = Pipeline([('pre', preprocessor), ('xgb', xgb_clf)])
        pipe_xgb.fit(X_train, y_train)
        preds_proba = pipe_xgb.predict_proba(X_val)[:, 1]
        preds = pipe_xgb.predict(X_val)
        results['xgb'] = {
            'auc': roc_auc_score(y_val, preds_proba),
            'accuracy': accuracy_score(y_val, preds),
            'model': pipe_xgb
        }
        joblib.dump(pipe_xgb, model_dir / 'clf_xgb.pkl')

    if HAS_CAT and not FORCE_NO_CATBOOST:
        try:
            cat_cols = X_train.select_dtypes(
                include=['object', 'category']).columns.tolist()
            # Convert NaN to string for categorical columns
            X_train_cat = X_train.copy()
            X_val_cat = X_val.copy()
            for col in cat_cols:
                X_train_cat[col] = X_train_cat[col].astype(
                    str).replace('nan', 'missing')
                X_val_cat[col] = X_val_cat[col].astype(
                    str).replace('nan', 'missing')

            logger.info('Training CatBoostClassifier...')
            cbc = CatBoostClassifier(
                iterations=500, learning_rate=0.05, random_seed=RANDOM_STATE, verbose=0)
            cbc.fit(X_train_cat, y_train, cat_features=cat_cols)
            preds_proba = cbc.predict_proba(X_val_cat)[:, 1]
            preds = cbc.predict(X_val_cat)
            results['catboost'] = {
                'auc': roc_auc_score(y_val, preds_proba),
                'accuracy': accuracy_score(y_val, preds),
                'model': cbc
            }
            joblib.dump(cbc, model_dir / 'clf_catboost.pkl')
        except (CatBoostError, ValueError, TypeError) as e:
            logger.warning(f"CatBoost classifier failed: {e}")

    for name, res in results.items():
        logger.info(
            f"Classifier {name}: AUC={res['auc']:.4f}, Acc={res['accuracy']:.4f}")

    return results


# -----------------------
# SHAP analysis
# -----------------------

def run_shap_analysis(model, X_sample: pd.DataFrame, model_type: str = 'regression', nsamples: int = 1000):
    if not HAS_SHAP:
        logger.warning('SHAP not installed. Skipping SHAP analysis.')
        return None

    logger.info('Running SHAP analysis...')
    # If model is a pipeline, extract final estimator
    if isinstance(model, Pipeline):
        pre = model.named_steps.get('pre')
        estimator = list(model.named_steps.values())[-1]
        X_trans = pre.transform(X_sample)
    else:
        estimator = model
        X_trans = X_sample

    explainer = None
    try:
        explainer = shap.Explainer(estimator)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample)
        return shap_values
    except Exception as e:
        logger.warning(f'SHAP analysis failed: {e}')
        return None


# -----------------------
# Orchestrator
# -----------------------

def run_pipeline(data_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    # Ensure the prepared dataset contains the columns we rely on
    if 'TotalClaims' not in df.columns:
        raise RuntimeError('TotalClaims not found in dataset')

    # 1) Claim severity model (only rows with TotalClaims > 0)
    df_sev = df[df['TotalClaims'] > 0].copy()
    logger.info(f"Claim severity dataset size: {len(df_sev):,}")

    # Select features (drop identifiers)
    df_sev = select_features(df_sev, drop_cols=['PolicyID'])

    # Create stratify bin for stable regression split
    df_sev['StratifyBin'] = pd.qcut(
        df_sev['Log_TotalClaims'], q=5, labels=False, duplicates='drop')

    # Split
    X = df_sev.drop(columns=['TotalClaims', 'Log_TotalClaims', 'Sqrt_TotalClaims',
                    'Std_TotalClaims', 'HighClaim', 'ClaimSeverityCategory', 'StratifyBin'], errors='ignore')
    y = df_sev['Log_TotalClaims']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=df_sev['StratifyBin'])

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    reg_models = train_regression_models(
        X_train, y_train, X_val, y_val, preprocessor, out_dir / 'regression')

    # 2) Claim probability model (classification) - model HighClaim (binary)
    df_clf = df.copy()
    df_clf = select_features(df_clf, drop_cols=['PolicyID'])

    # target
    if 'HighClaim' not in df_clf.columns:
        raise RuntimeError(
            'HighClaim target not found. Run preparer to create target columns.')

    Xc = df_clf.drop(columns=['TotalClaims', 'Log_TotalClaims', 'Sqrt_TotalClaims',
                     'Std_TotalClaims', 'HighClaim', 'ClaimSeverityCategory'], errors='ignore')
    yc = df_clf['HighClaim']

    Xc_train, Xc_val, yc_train, yc_val = train_test_split(
        Xc, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc)

    preprocessor_clf, num_cols_c, cat_cols_c = build_preprocessor(Xc_train)

    clf_models = train_classification_models(
        Xc_train, yc_train, Xc_val, yc_val, preprocessor_clf, out_dir / 'classification')

    # 3) Basic premium calculator example (naive pricing)
    best_clf = clf_models.get('catboost') or clf_models.get(
        'xgb') or clf_models.get('rf') or clf_models.get('logistic')
    best_reg = reg_models.get('catboost') or reg_models.get(
        'xgb') or reg_models.get('rf') or reg_models.get('linear')

    pricing = {
        'notes': 'Naive risk-based premium example',
        'formula': 'Premium = P(claim) * E[severity] + expense_loading + profit_margin'
    }

    with open(out_dir / 'model_metadata.json', 'w') as f:
        json.dump({'regression': list(reg_models.keys()), 'classification': list(
            clf_models.keys()), 'pricing_example': pricing}, f, indent=2)

    logger.info('Modeling pipeline finished.')


# -----------------------
# CLI
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run modeling pipeline')
    parser.add_argument(
        '--data', type=str, default='./data/processed/claim_data_prepared.csv', help='Path to prepared CSV')
    parser.add_argument('--out', type=str, default='./models',
                        help='Output directory for models')

    args = parser.parse_args()
    run_pipeline(args.data, args.out)
