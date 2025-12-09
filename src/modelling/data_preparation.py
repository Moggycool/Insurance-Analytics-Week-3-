import pandas as pd
import numpy as np

file_path = r"D:\Python\Week-3\Raw_Data\MachineLearningRating_v3.txt"
output_path = r"D:\Python\Week-3\Processed_Data\claim_policies_subset.csv"

print("Extracting claim policies using optimized approach...")

# Define which columns we need for modeling
# Based on your earlier analysis, these seem important
essential_columns = [
    'PolicyID', 'TransactionMonth', 'Country', 'Province', 'MainCrestaZone',
    'VehicleType', 'make', 'Model', 'RegistrationYear', 'cubiccapacity',
    'kilowatts', 'bodytype', 'AlarmImmobiliser', 'TrackingDevice',
    'CapitalOutstanding', 'SumInsured', 'TotalPremium', 'ExcessSelected',
    'CoverType', 'Product', 'Gender', 'NumberOfVehiclesInFleet',
    'TotalClaims'  # Target variable
]

chunksize = 100000
claim_chunks = []
total_rows_processed = 0

print("Processing data in chunks...")
for chunk in pd.read_csv(file_path, sep='|', usecols=essential_columns,
                         chunksize=chunksize, low_memory=False):
    total_rows_processed += len(chunk)

    # Filter for claims
    claim_chunk = chunk[chunk['TotalClaims'] > 0]
    if len(claim_chunk) > 0:
        claim_chunks.append(claim_chunk)
        print(
            f"Processed {total_rows_processed:,} rows | Found {len(claim_chunk)} claim(s) in this chunk")

    # Progress indicator
    if total_rows_processed % 500000 == 0:
        print(f"Progress: {total_rows_processed:,} rows processed")

# Combine all claim chunks
if claim_chunks:
    df_claims = pd.concat(claim_chunks, ignore_index=True)
    print(f"\n✅ Successfully extracted {len(df_claims)} claim records")

    # Save for modeling
    df_claims.to_csv(output_path, index=False)
    print(f"📁 Saved to: {output_path}")

    # Basic statistics
    print("\n📊 Claim Statistics:")
    print(f"Total claim amount: R{df_claims['TotalClaims'].sum():,.2f}")
    print(f"Average claim: R{df_claims['TotalClaims'].mean():,.2f}")
    print(f"Median claim: R{df_claims['TotalClaims'].median():,.2f}")
    print(f"Std deviation: R{df_claims['TotalClaims'].std():,.2f}")

    # Check data quality
    print("\n🔍 Data Quality Check:")
    print(
        f"Missing values in TotalClaims: {df_claims['TotalClaims'].isnull().sum()}")
    print(
        f"Zero claims (should be 0): {(df_claims['TotalClaims'] == 0).sum()}")

    # Show sample
    print("\n📋 Sample of claim data:")
    print(df_claims.head())

else:
    print("❌ No claims found in the dataset!")
