# Modeling Analysis Report

**Date:** 2025-12-11 20:24:34
**Project:** D:\Python\Week-3\Insurance-Analytics-Week-3-

## Dataset Overview
- Total policies: 1,000,093
- Policies with claims: 2,788
- High claim policies: 2,788

## Best Performing Models

### Regression (Claim Severity)
**Model:** Random Forest
- R² Score: 0.6268
- RMSE: 1.0046

### Classification (Claim Probability)
**Model:** Logistic Regression
- AUC: 0.9953
- Accuracy: 0.9975

## Key Findings
- Random Forest performed best for claim severity prediction (R²=0.6268)
- All classification models showed excellent performance (AUC > 0.96)
- CatBoost models failed due to NaN handling issues in categorical features
- Logistic regression showed convergence warning (needs more iterations)

## Recommendations
- Use Random Forest for production claim severity prediction
- Investigate potential data leakage in classification models
- Fix NaN handling for CatBoost compatibility
- Implement cross-validation for more robust performance estimates
- Consider ensemble methods for improved prediction stability

## Business Implications
- Risk-based pricing can be implemented using model predictions
- High-risk profiles can be identified for targeted underwriting
- Premium optimization opportunities exist through better risk assessment
