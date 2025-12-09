import pandas as pd

file_path = r"D:\Python\Week-3\Raw_Data\MachineLearningRating_v3.txt"

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
