"""
Visualization helpers for Insurance Analytics
Converted to class-based design
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class DataVisualizer:
    """Visualization class for all EDA-related plots."""

    @staticmethod
    def plot_histogram(df: pd.DataFrame, column: str, bins: int = 50, title: str = None):
        """Plot histogram with KDE."""
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column].dropna(), bins=bins, kde=True, color="skyblue")
        plt.title(title or f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    @staticmethod
    def plot_boxplot(df: pd.DataFrame, column: str, by: str = None, title: str = None):
        """Plot a boxplot with optional grouping."""
        data = df[column].dropna()

        if data.nunique() <= 1:
            print(f"Skipping boxplot for '{column}' — insufficient variance.")
            return

        plt.figure(figsize=(12, 6))

        if by and by in df.columns:
            sns.boxplot(data=df, x=by, y=column)
        else:
            sns.boxplot(data=df, y=column)

        plt.title(title or f"Boxplot of {column}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, cols: list = None, title: str = "Correlation Heatmap"):
        """Correlation heatmap for numeric columns."""
        numeric_cols = cols or df.select_dtypes(
            include=np.number).columns.tolist()
        corr = df[numeric_cols].corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_loss_ratio_vs_premium(
        df: pd.DataFrame,
        premium_col: str = "TotalPremium",
        loss_ratio_col: str = "LossRatio"
    ):
        """Scatter plot: Loss Ratio vs Premium."""
        if premium_col not in df.columns or loss_ratio_col not in df.columns:
            raise ValueError("Columns not found in DataFrame")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=premium_col, y=loss_ratio_col, data=df, alpha=0.5)
        plt.xscale("log")
        plt.title("Loss Ratio vs Premium")
        plt.xlabel("Total Premium (log scale)")
        plt.ylabel("Loss Ratio")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_importances: pd.Series, top_n: int = 20,
                                title: str = "Feature Importance"):
        """Horizontal bar plot for feature importance."""
        top_features = feature_importances.sort_values(
            ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_features.values,
                    y=top_features.index, palette="viridis")
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
