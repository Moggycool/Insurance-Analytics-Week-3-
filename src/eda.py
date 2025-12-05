"""
Exploratory Data Analysis (EDA) for Insurance Analytics
Aligned with DataPreprocessor, utils.py, and visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import summarize_dataframe, memory_usage
from src.visualization import plot_histogram, plot_boxplot, plot_correlation_heatmap, plot_loss_ratio_vs_premium

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})


class InsuranceEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        self.cat_cols = self.df.select_dtypes(
            include="category").columns.tolist()
        self.date_cols = self.df.select_dtypes(
            include="datetime").columns.tolist()

    # -----------------------
    # 1) Data Summarization
    # -----------------------
    def data_structure_summary(self):
        print("\n" + "="*60)
        print("DATA STRUCTURE & DTYPE CHECK")
        print("="*60)
        print(self.df.dtypes)
        print("\n" + "="*60)
        print(f"Memory usage: {memory_usage(self.df):.2f} MB")
        print(f"Dataset shape: {self.df.shape}")

    def descriptive_statistics(self):
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS (Numerical Features)")
        print("="*60)
        print(self.df[self.numeric_cols].describe().T)
        print("\n" + "="*60)

    # -----------------------
    # 2) Data Quality Assessment
    # -----------------------
    def missing_value_summary(self):
        print("\n" + "="*60)
        print("MISSING VALUE ASSESSMENT")
        print("="*60)
        missing = self.df.isna().sum()
        missing_pct = missing / len(self.df) * 100
        missing_df = pd.DataFrame({"Missing": missing, "Percent": missing_pct})
        print(missing_df[missing_df["Missing"] > 0])

    # -----------------------
    # 3) Univariate Analysis
    # -----------------------
    def plot_univariate_distributions(self):
        print("\nPlotting numerical distributions...")
        for col in self.numeric_cols:
            plot_histogram(self.df, col)
        print("\nPlotting categorical distributions...")
        for col in self.cat_cols:
            plt.figure(figsize=(12, 6))
            self.df[col].value_counts().plot(kind="bar", color="skyblue")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()

    # -----------------------
    # 4) Bivariate / Multivariate Analysis
    # -----------------------
    def correlation_analysis(self):
        print("\nPlotting correlation heatmap for numeric variables...")
        plot_correlation_heatmap(self.df, self.numeric_cols)

    def monthly_change_scatter(self, col1="TotalPremium", col2="TotalClaims", group_by="PostalCode"):
        if col1 not in self.df.columns or col2 not in self.df.columns:
            print(f"Columns {col1} or {col2} not found.")
            return

        if group_by not in self.df.columns:
            print(f"Grouping column {group_by} not found.")
            return

        # Aggregate by group and month
        if "TransactionMonth" in self.df.columns:
            df_grouped = self.df.groupby([group_by, pd.Grouper(key="TransactionMonth", freq="M")])[
                [col1, col2]].sum().reset_index()
            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=col1, y=col2, hue=group_by,
                            data=df_grouped, alpha=0.7)
            plt.title(f"Monthly {col1} vs {col2} by {group_by}")
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.show()

    # -----------------------
    # 5) Data Comparison Over Geography / Category
    # -----------------------
    def compare_trends(self, feature="CoverType", value_col="TotalPremium"):
        if feature not in self.df.columns or value_col not in self.df.columns:
            print(f"Columns {feature} or {value_col} not found.")
            return

        trend_df = self.df.groupby([feature, pd.Grouper(key="TransactionMonth", freq="M")])[
            value_col].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="TransactionMonth", y=value_col,
                     hue=feature, data=trend_df, marker="o")
        plt.title(f"Trend of {value_col} by {feature} over time")
        plt.xlabel("Month")
        plt.ylabel(value_col)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()

    # -----------------------
    # 6) Outlier Detection
    # -----------------------
    def plot_outliers(self):
        for col in self.numeric_cols:
            series = self.df[col].dropna()

            # Skip if column is empty or constant
            if series.nunique() <= 1:
                print(
                    f"Skipping '{col}' — not enough unique values for a boxplot.")
                continue

            # Skip if column cannot be converted to float
            try:
                series = pd.to_numeric(series)
            except:
                print(f"Skipping '{col}' — non-numeric values detected.")
                continue

            # Safe call to the boxplot
            plot_boxplot(self.df, col)

    # -----------------------
    # 7) Creative / Insightful Visualizations
    # -----------------------

    def creative_visualizations(self):
        # Example 1: Loss Ratio vs Premium (scatter with log scale)
        if "LossRatio" in self.df.columns and "TotalPremium" in self.df.columns:
            plot_loss_ratio_vs_premium(self.df)

        # Example 2: Premium per Cover Type (bar chart)
        if "CoverType" in self.df.columns and "TotalPremium" in self.df.columns:
            cover_premium = self.df.groupby(
                "CoverType")["TotalPremium"].sum().sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=cover_premium.index,
                        y=cover_premium.values, palette="magma")
            plt.title("Total Premium by Cover Type")
            plt.xlabel("Cover Type")
            plt.ylabel("Total Premium")
            plt.xticks(rotation=45)
            plt.show()

        # Example 3: Vehicle Age vs Loss Ratio (scatter with trend)
        if "VehicleAge" in self.df.columns and "LossRatio" in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x="VehicleAge", y="LossRatio",
                            data=self.df, alpha=0.5)
            sns.regplot(x="VehicleAge", y="LossRatio",
                        data=self.df, scatter=False, color="red")
            plt.title("Vehicle Age vs Loss Ratio")
            plt.xlabel("Vehicle Age")
            plt.ylabel("Loss Ratio")
            plt.show()
