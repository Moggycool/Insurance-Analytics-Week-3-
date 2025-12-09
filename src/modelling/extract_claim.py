""" A module to Extract Claim"""
import pandas as pd
import numpy as np
import os
import sys

file_path = r"D:\Python\Week-3\Raw_Data\MachineLearningRating_v3.txt"
output_path = r"D:\Python\Week-3\Insurance-Analytics-Week-3-\data\processed\claim_policies_subset.csv"

# Quick analysis of TotalClaims distribution
print("Analyzing TotalClaims distribution...")

# Read only the TotalClaims column
df_claims = pd.read_csv(file_path, sep='|', usecols=['TotalClaims'])

print(f"Total rows: {len(df_claims):,}")
print(f"Rows with TotalClaims > 0: {(df_claims['TotalClaims'] > 0).sum():,}")
print(
    f"Percentage with claims: {(df_claims['TotalClaims'] > 0).sum()/len(df_claims)*100:.2f}%")

# Distribution of claim amounts
print("\nClaim amount statistics for rows with claims:")
claims_positive = df_claims[df_claims['TotalClaims'] > 0]['TotalClaims']
print(f"Min claim: {claims_positive.min():,.2f}")
print(f"Max claim: {claims_positive.max():,.2f}")
print(f"Mean claim: {claims_positive.mean():,.2f}")
print(f"Median claim: {claims_positive.median():,.2f}")
# Since you only have 2,788 claims, you can load them directly
print("Extracting claim policies for severity modeling...")

# Read the entire dataset (since 1M rows is manageable)
df = pd.read_csv(file_path, sep='|')

# Create claim subset
df_claims = df[df['TotalClaims'] > 0].copy()
print(f"Extracted {len(df_claims)} claim records")

# Save for modeling
df_claims.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

# Summary of claim amounts
print("\nClaim Severity Distribution:")
print(f"Total claims cost: R{df_claims['TotalClaims'].sum():,.2f}")
print(f"Average claim: R{df_claims['TotalClaims'].mean():,.2f}")
print(f"Std deviation: R{df_claims['TotalClaims'].std():,.2f}")
