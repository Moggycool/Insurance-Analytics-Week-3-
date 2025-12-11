# Insurance Analytics - Week 3 EDA

This repository contains a **comprehensive Exploratory Data Analysis (EDA)** for insurance analytics, designed to meet all Week 3 rubric requirements. The notebook `eda_analysis.ipynb` guides you through data loading, preprocessing, univariate and multivariate analyses, outlier detection, and additional insights.

---

## Project Overview
This project analyzes historical insurance claim data for AlphaCare Insurance Solutions to optimize marketing strategy and identify "low-risk" client segments in South Africa.

## 📂 Project Structure

```
Insurance-Analytics-Week-3-
├─ .dvc
│  └─ config
├─ .dvcignore
├─ data
│  └─ output
├─ dvc.lock
├─ dvc.yaml
├─ notebooks
│  └─ eda_analysis.ipynb
├─ README.md
├─ reports
│  ├─ correlation_matrix.csv
│  ├─ data_sample.csv
│  ├─ descriptive_statistics.csv
│  ├─ eda_summary.txt
│  └─ outlier_report.json
├─ requirements.txt
├─ src
│  ├─ data_preprocessing.py
│  ├─ eda.py
│  ├─ utils.py
│  ├─ visualization.py
│  └─ __init__.py
└─ tests

```


## Setup Instructions
1. Clone the repository: `git clone https://github.com/Moggycool/Insurance-Analytics-Week-3-.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Jupyter notebooks in the notebooks/ directory

## Install dependencies:
- pip install -r requirements.txt

## Ensure DVC is installed (for data versioning):
- pip install dvc

## Pull raw and processed data with DVC:
- dvc pull

# 📝 Notebook Overview
1. Initial Setup & Imports
- Imports essential libraries: pandas, numpy, matplotlib, seaborn.
- Sets up visual styles for plots.
- Imports custom modules: DataPreprocessor, DataUtils, DataVisualizer.

2. Rubric Assessment Tracker
- Tracks Week 3 rubric compliance.
- Rubrics covered include:
- Data summarization
- Data quality assessment
- Univariate and multivariate analysis
- Outlier detection
- DVC, Git, and project structure checks

3. Data Loading & Preprocessing
- Loads processed data if available; otherwise, applies preprocessing.
- Handles feature creation, log transformations, and data quality checks.
- Supports large datasets with chunked processing.

4. Exploratory Data Analysis
- Data Summarization:
- Dataset dimensions, memory usage, data types.
- Descriptive statistics including skewness, kurtosis, and zero counts.
- Saved to reports/descriptive_statistics.csv.
- Data Quality Assessment:
- Missing value analysis.
- Duplicate detection.
- Invalid value checks for key columns.
- Univariate Analysis:
- Histograms for numerical features.
- Bar charts for categorical features.
- Bivariate / Multivariate Analysis:
- Correlation matrix heatmap.
- Scatter plots with trend lines.
- Saved correlation matrix to reports/correlation_matrix.csv.
- Outlier Detection:
- Boxplots with IQR method.
- Outlier summary saved to reports/outlier_report.json.

5. Additional Analyses & Insights
- Time Series Analysis (if TransactionMonth exists): Monthly premiums, claims, policy counts, and loss ratio.
- Categorical Analysis: Analysis of CoverType with summary statistics and visualizations.

6. Reports & Outputs
- Dataset sample: reports/data_sample.csv
- EDA summary report: reports/eda_summary.txt
- Descriptive statistics: reports/descriptive_statistics.csv
- Correlation matrix: reports/correlation_matrix.csv
- Outlier report: reports/outlier_report.json

# 🔜 Next Steps
- Verify DVC pipeline for raw and processed data.
- Ensure Git practices are documented (commits, branches, pull requests) as required.
- Review and assure repository structure for compliance with Rubric 4.
- Expand on statistical analyses in statistical_analysis.ipynb. and Interprate Results.

## Prepared by
- Moges Behailu





```
Insurance-Analytics-Week-3-
├─ .dvc
│  └─ config
├─ .dvcignore
├─ catboost_info
│  ├─ catboost_training.json
│  ├─ learn
│  │  └─ events.out.tfevents
│  ├─ learn_error.tsv
│  ├─ time_left.tsv
│  └─ tmp
├─ data
│  ├─ hypothesis_results
│  │  └─ visualizations
│  │     ├─ categorical_distributions.png
│  │     └─ numerical_distributions.png
│  └─ output
├─ dvc.lock
├─ dvc.yaml
├─ modeling_pipeline.log
├─ models
│  ├─ best_Lasso.pkl
│  ├─ best_LinearRegression.pkl
│  ├─ classification
│  │  ├─ clf_logistic.pkl
│  │  ├─ clf_rf.pkl
│  │  ├─ clf_xgb.pkl
│  │  └─ reg_catboost.pkl
│  ├─ cross_validation_results.json
│  ├─ DecisionTree.pkl
│  ├─ ElasticNet.pkl
│  ├─ GradientBoosting.pkl
│  ├─ Lasso.pkl
│  ├─ Lasso_best_params.json
│  ├─ Lasso_tuned.pkl
│  ├─ LightGBM.pkl
│  ├─ LinearRegression.pkl
│  ├─ LinearRegression_best_params.json
│  ├─ LinearRegression_tuned.pkl
│  ├─ model_comparison.json
│  ├─ model_comparison_combined.png
│  ├─ model_comparison_r2.png
│  ├─ model_metadata.json
│  ├─ preprocessor.pkl
│  ├─ RandomForest.pkl
│  ├─ regression
│  │  ├─ reg_linear.pkl
│  │  ├─ reg_rf.pkl
│  │  └─ reg_xgb.pkl
│  ├─ Ridge.pkl
│  └─ XGBoost.pkl
├─ notebooks
│  ├─ eda_analysis.ipynb
│  ├─ hypothesis_eda.ipynb
│  ├─ modeling.ipynb
│  └─ reports
├─ Processed_Data
├─ README.md
├─ reports
│  ├─ eda_summary.txt
│  └─ outlier_report.json
├─ requirements.txt
├─ Results
│  ├─ modeling_eda_summary.json
│  ├─ model_diagnostic_report.json
│  └─ Task4_Reports
│     ├─ SHAP_Analysis
│     ├─ task4_comprehensive_report.json
│     └─ task4_final_report.md
├─ src
│  ├─ data_preprocessing.py
│  ├─ eda.py
│  ├─ hypothesis_testing
│  │  ├─ data_loader.py
│  │  ├─ hypothesis_evaluator.py
│  │  ├─ metrics.py
│  │  ├─ report_generator.py
│  │  ├─ runner.py
│  │  ├─ run_hypothesis.py
│  │  ├─ segmentation.py
│  │  ├─ statistical_tests.py
│  │  └─ __init__.py
│  ├─ hypothesis_testing.py
│  ├─ modelling
│  │  ├─ data_preparation.py
│  │  ├─ extract_claim.py
│  │  ├─ featuring.py
│  │  ├─ main.py
│  │  ├─ modeling.py
│  │  └─ prep.py
│  ├─ utils.py
│  ├─ visualization.py
│  └─ __init__.py
└─ tests

```