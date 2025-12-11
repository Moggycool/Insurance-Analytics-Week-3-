# SHAP Analysis Report

## Summary
- Analysis Date: 2025-12-11 21:53:24
- SHAP Version: 0.50.0
- Models Analyzed: Regression only

## Key Findings

### Regression Model (Claim Severity)
- Top 5 Features: num__Log_SumInsured, num__SumInsured, ord__CoverCategory, num__PostalCode, num__Log_CalculatedPremiumPerTerm
- Base Value: 8.9231
- Features Analyzed: 112

### Classification Model (Claim Probability)

## Generated Files
The following files were generated:
- shap_regression_analysis.json
- shap_regression_dependence.png
- shap_regression_importance.png
- shap_regression_summary.png
- shap_regression_top_features.csv

## Business Implications
1. Use SHAP feature importance for risk assessment prioritization
2. Monitor top features for model drift detection
3. Validate underwriting rules against SHAP explanations
4. Use insights for feature engineering in next model iteration
