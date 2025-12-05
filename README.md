# ACIS Insurance Analytics Project

## Project Overview
This project analyzes historical insurance claim data for AlphaCare Insurance Solutions to optimize marketing strategy and identify "low-risk" client segments in South Africa.

## Business Objective
Develop predictive analytics to identify low-risk insurance targets for premium reduction strategies, helping attract new clients through targeted marketing.

## Repository Structure

```
Insurance-Analytics-Week-3-
├─ .dvc
│  ├─ cache
│  │  └─ files
│  │     └─ md5
│  │        └─ 53
│  │           └─ f65fc5b8ea785e5fb22b1c6113a7a7
│  ├─ config
│  └─ tmp
│     ├─ btime
│     ├─ lock
│     ├─ rwlock
│     └─ rwlock.lock
├─ .dvcignore
├─ data
│  ├─ output
│  ├─ processed
│  └─ raw
│     ├─ sample_raw.txt
│     └─ sample_raw.txt.dvc
├─ notebooks
│  ├─ eda_analysis.ipynb
│  ├─ src
│  │  └─ data_preprocessing.py
│  └─ statistical_analysis.ipynb
├─ README.md
├─ requirements.txt
├─ src
│  ├─ data_preprocessing.py
│  ├─ utils.py
│  ├─ visualization.py
│  └─ __init__.py
└─ tests

```


## Setup Instructions
1. Clone the repository: `git clone https://github.com/Moggycool/Insurance-Analytics-Week-3-.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Jupyter notebooks in the notebooks/ directory

## Key Analyses
- Loss Ratio analysis by Province, VehicleType, Gender
- Temporal trends in claims over 18 months
- Vehicle make/model risk profiling
- Outlier detection in claims and vehicle values

## Team
Data Analytics Team - AlphaCare Insurance Solutions

