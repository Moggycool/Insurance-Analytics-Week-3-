"""
Module for extracting claim records from the raw insurance data file.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClaimDataExtractor:
    """
    Extracts claim records from the raw insurance data file using memory-efficient chunking.
    """

    def __init__(self, raw_data_path: str, output_dir: str = None):
        """
        Initialize the claim data extractor.

        Args:
            raw_data_path: Path to the raw data file
            output_dir: Directory to save extracted data
        """
        self.raw_data_path = Path(raw_data_path)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to processed data directory
            self.output_dir = self.raw_data_path.parent.parent / 'Processed_Data'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Essential columns based on your data
        self.essential_columns = [
            'PolicyID', 'TransactionMonth', 'Gender', 'Country', 'Province',
            'MainCrestaZone', 'VehicleType', 'RegistrationYear', 'make', 'Model',
            'cubiccapacity', 'kilowatts', 'bodytype', 'AlarmImmobiliser',
            'TrackingDevice', 'CapitalOutstanding', 'NumberOfVehiclesInFleet',
            'SumInsured', 'ExcessSelected', 'CoverType', 'Product',
            'TotalPremium', 'TotalClaims'  # Target variable
        ]

        logger.info(f"Initialized ClaimDataExtractor")
        logger.info(f"Raw data path: {self.raw_data_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def _clean_numeric_string(self, value):
        """Clean numeric strings by removing commas and converting to float."""
        if isinstance(value, str):
            # Remove any thousand separators (commas)
            value = value.replace(',', '')
            # Replace European decimal comma with dot if present
            if '.' in value and ',' in value:
                # If both are present, assume comma is thousand separator
                value = value.replace(',', '')
            elif ',' in value and value.count(',') == 1:
                # If single comma, assume it's decimal separator
                value = value.replace(',', '.')

        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    def analyze_claim_distribution(self, chunksize: int = 100000) -> dict:
        """
        Analyze the distribution of TotalClaims in the dataset.

        Args:
            chunksize: Number of rows to read at once

        Returns:
            Dictionary with claim statistics
        """
        logger.info("Analyzing TotalClaims distribution...")

        total_rows = 0
        claim_count = 0
        claim_amounts = []

        for chunk in pd.read_csv(
            self.raw_data_path,
            sep='|',
            usecols=['TotalClaims'],
            chunksize=chunksize,
            low_memory=False,
            dtype={'TotalClaims': 'object'}  # Read as object first
        ):
            total_rows += len(chunk)

            # Clean the TotalClaims column
            chunk['TotalClaims'] = chunk['TotalClaims'].apply(
                self._clean_numeric_string)

            chunk_claims = chunk[chunk['TotalClaims'] > 0]
            claim_count += len(chunk_claims)

            if len(chunk_claims) > 0:
                claim_amounts.extend(chunk_claims['TotalClaims'].tolist())

        # Calculate statistics
        claim_amounts = np.array(claim_amounts)
        stats = {
            'total_rows': total_rows,
            'claim_count': claim_count,
            'claim_percentage': (claim_count / total_rows * 100) if total_rows > 0 else 0,
            'min_claim': float(np.min(claim_amounts)) if len(claim_amounts) > 0 else 0,
            'max_claim': float(np.max(claim_amounts)) if len(claim_amounts) > 0 else 0,
            'mean_claim': float(np.mean(claim_amounts)) if len(claim_amounts) > 0 else 0,
            'median_claim': float(np.median(claim_amounts)) if len(claim_amounts) > 0 else 0,
            'std_claim': float(np.std(claim_amounts)) if len(claim_amounts) > 0 else 0
        }

        # Print summary
        logger.info(f"Total rows: {stats['total_rows']:,}")
        logger.info(f"Rows with TotalClaims > 0: {stats['claim_count']:,}")
        logger.info(
            f"Percentage with claims: {stats['claim_percentage']:.2f}%")

        if stats['claim_count'] > 0:
            logger.info("\nClaim amount statistics for rows with claims:")
            logger.info(f"Min claim: R{stats['min_claim']:,.2f}")
            logger.info(f"Max claim: R{stats['max_claim']:,.2f}")
            logger.info(f"Mean claim: R{stats['mean_claim']:,.2f}")
            logger.info(f"Median claim: R{stats['median_claim']:,.2f}")
            logger.info(f"Std deviation: R{stats['std_claim']:,.2f}")

        return stats

    def extract_claim_records(self, chunksize: int = 50000) -> pd.DataFrame:
        """
        Extract all records with TotalClaims > 0 using memory-efficient chunking.

        Args:
            chunksize: Number of rows to read at once

        Returns:
            DataFrame containing only claim records
        """
        logger.info("Extracting claim policies for severity modeling...")

        claim_chunks = []
        total_rows_processed = 0
        total_claims_found = 0

        try:
            # First, analyze to know what we're dealing with
            stats = self.analyze_claim_distribution()

            if stats['claim_count'] == 0:
                logger.warning("No claims found in the dataset!")
                return pd.DataFrame()

            logger.info(
                f"\nStarting extraction of {stats['claim_count']:,} claim records...")

            # Read data in chunks - as objects first to handle mixed types
            for i, chunk in enumerate(pd.read_csv(
                self.raw_data_path,
                sep='|',
                usecols=self.essential_columns,
                chunksize=chunksize,
                low_memory=False,
                # Read all as object first
                dtype={col: 'object' for col in self.essential_columns}
            )):
                total_rows_processed += len(chunk)

                # Clean numeric columns
                numeric_columns = ['RegistrationYear', 'cubiccapacity', 'kilowatts',
                                   'SumInsured', 'TotalPremium', 'CapitalOutstanding',
                                   'TotalClaims', 'ExcessSelected', 'NumberOfVehiclesInFleet']

                for col in numeric_columns:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].apply(
                            self._clean_numeric_string)

                # Filter for claims
                claim_chunk = chunk[chunk['TotalClaims'] > 0].copy()

                if len(claim_chunk) > 0:
                    claim_chunks.append(claim_chunk)
                    total_claims_found += len(claim_chunk)
                    logger.info(
                        f"Chunk {i+1}: Found {len(claim_chunk)} claim(s)")

                # Progress update
                if (i + 1) % 20 == 0:
                    logger.info(
                        f"Progress: {total_rows_processed:,} rows processed")

            # Combine all claim chunks
            if claim_chunks:
                df_claims = pd.concat(claim_chunks, ignore_index=True)
                logger.info(
                    f"Successfully extracted {len(df_claims)} claim records")

                # Convert numeric columns to appropriate types
                self._convert_column_types(df_claims)

                # Save extracted data
                self._save_claim_data(df_claims)

                # Generate summary statistics
                self._generate_summary(df_claims)

                return df_claims
            else:
                logger.warning("No claims found during extraction!")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error extracting claim records: {str(e)}")
            raise

    def _convert_column_types(self, df: pd.DataFrame) -> None:
        """Convert columns to appropriate data types."""
        # Convert to appropriate types
        if 'RegistrationYear' in df.columns:
            df['RegistrationYear'] = df['RegistrationYear'].astype('float64')

        if 'TotalClaims' in df.columns:
            df['TotalClaims'] = df['TotalClaims'].astype('float64')

        if 'TotalPremium' in df.columns:
            df['TotalPremium'] = df['TotalPremium'].astype('float64')

        if 'SumInsured' in df.columns:
            df['SumInsured'] = df['SumInsured'].astype('float64')

        if 'cubiccapacity' in df.columns:
            df['cubiccapacity'] = df['cubiccapacity'].astype('float64')

        if 'kilowatts' in df.columns:
            df['kilowatts'] = df['kilowatts'].astype('float64')

    def _save_claim_data(self, df_claims: pd.DataFrame) -> None:
        """Save extracted claim data to CSV file."""
        output_path = self.output_dir / 'claim_policies.csv'
        df_claims.to_csv(output_path, index=False)
        logger.info(f"Saved claim data to: {output_path}")

        # Also save a sample for quick inspection
        sample_path = self.output_dir / 'claim_policies_sample.csv'
        df_claims.sample(min(1000, len(df_claims))).to_csv(
            sample_path, index=False)
        logger.info(f"Saved sample data to: {sample_path}")

        # Save column information
        col_info_path = self.output_dir / 'column_info.txt'
        with open(col_info_path, 'w') as f:
            f.write("Column Information for Claim Data\n")
            f.write("="*50 + "\n")
            for col in df_claims.columns:
                f.write(f"{col}: {df_claims[col].dtype}\n")
                f.write(f"  Unique values: {df_claims[col].nunique()}\n")
                f.write(f"  Missing: {df_claims[col].isnull().sum()}\n")
                if df_claims[col].dtype in ['float64', 'int64']:
                    f.write(f"  Min: {df_claims[col].min()}\n")
                    f.write(f"  Max: {df_claims[col].max()}\n")
                f.write("\n")
        logger.info(f"Saved column information to: {col_info_path}")

    def _generate_summary(self, df_claims: pd.DataFrame) -> None:
        """Generate and log summary statistics."""
        logger.info("\n" + "="*50)
        logger.info("CLAIM DATA SUMMARY")
        logger.info("="*50)
        logger.info(f"Total claims extracted: {len(df_claims):,}")

        # Claim amount statistics
        total_claim_amount = df_claims['TotalClaims'].sum()
        avg_claim = df_claims['TotalClaims'].mean()
        median_claim = df_claims['TotalClaims'].median()
        std_claim = df_claims['TotalClaims'].std()

        logger.info(f"Total claim amount: R{total_claim_amount:,.2f}")
        logger.info(f"Average claim: R{avg_claim:,.2f}")
        logger.info(f"Median claim: R{median_claim:,.2f}")
        logger.info(f"Std deviation: R{std_claim:,.2f}")
        logger.info(f"Minimum claim: R{df_claims['TotalClaims'].min():,.2f}")
        logger.info(f"Maximum claim: R{df_claims['TotalClaims'].max():,.2f}")

        # Distribution statistics
        logger.info("\nClaim Distribution Percentiles:")
        percentiles = df_claims['TotalClaims'].quantile(
            [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        for pct, value in percentiles.items():
            logger.info(f"  {int(pct*100)}th percentile: R{value:,.2f}")

        # Premium statistics
        if 'TotalPremium' in df_claims.columns:
            logger.info("\nPremium Statistics:")
            logger.info(
                f"  Total premium: R{df_claims['TotalPremium'].sum():,.2f}")
            logger.info(
                f"  Average premium: R{df_claims['TotalPremium'].mean():,.2f}")
            logger.info(
                f"  Loss ratio (claims/premiums): {total_claim_amount/df_claims['TotalPremium'].sum()*100:.2f}%")

        # Vehicle type analysis
        if 'VehicleType' in df_claims.columns:
            logger.info("\nTop 5 Vehicle Types by Claim Count:")
            vehicle_counts = df_claims['VehicleType'].value_counts().head()
            for vehicle, count in vehicle_counts.items():
                logger.info(f"  {vehicle}: {count} claims")

    def extract_without_dtype_issues(self, chunksize: int = 50000) -> pd.DataFrame:
        """
        Alternative extraction method without specifying dtypes.
        Let pandas infer types and handle cleaning later.
        """
        logger.info("Using alternative extraction method...")

        claim_chunks = []

        for i, chunk in enumerate(pd.read_csv(
            self.raw_data_path,
            sep='|',
            chunksize=chunksize,
            low_memory=False,
            dtype='object'  # Read everything as object
        )):
            # Only keep essential columns
            chunk = chunk[self.essential_columns]

            # Clean TotalClaims column
            chunk['TotalClaims'] = chunk['TotalClaims'].apply(
                self._clean_numeric_string)

            # Filter for claims
            claim_chunk = chunk[chunk['TotalClaims'] > 0].copy()

            if len(claim_chunk) > 0:
                claim_chunks.append(claim_chunk)
                logger.info(f"Chunk {i+1}: Found {len(claim_chunk)} claim(s)")

        if claim_chunks:
            df_claims = pd.concat(claim_chunks, ignore_index=True)

            # Clean all numeric columns
            numeric_cols = ['RegistrationYear', 'cubiccapacity', 'kilowatts',
                            'SumInsured', 'TotalPremium', 'CapitalOutstanding',
                            'ExcessSelected', 'NumberOfVehiclesInFleet']

            for col in numeric_cols:
                if col in df_claims.columns:
                    df_claims[col] = df_claims[col].apply(
                        self._clean_numeric_string)

            self._convert_column_types(df_claims)

            # Save
            output_path = self.output_dir / 'claim_policies_alternative.csv'
            df_claims.to_csv(output_path, index=False)
            logger.info(f"Saved alternative extraction to: {output_path}")

            return df_claims

        return pd.DataFrame()


def main():
    """Main function to run the extractor."""
    # Define paths
    raw_data_path = r"D:\Python\Week-3\Insurance-Analytics-Week-3-\data\raw\MachineLearningRating_v3.txt"
    output_dir = r"D:\Python\Week-3\Insurance-Analytics-Week-3-\data\processed"

    # Initialize extractor
    extractor = ClaimDataExtractor(raw_data_path, output_dir)

    try:
        # Try the main extraction method
        logger.info("Attempting main extraction method...")
        df_claims = extractor.extract_claim_records()
    except Exception as e:
        logger.error(f"Main extraction failed: {str(e)}")
        logger.info("Trying alternative extraction method...")
        # Try alternative method
        df_claims = extractor.extract_without_dtype_issues()

    if len(df_claims) > 0:
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE!")
        print("="*60)
        print(f"Extracted {len(df_claims)} claim records")
        print(f"Saved to: {extractor.output_dir}")
        print("\nData Types:")
        print(df_claims.dtypes)
        print("\nFirst few rows:")
        print(df_claims.head())
    else:
        print("No claims extracted!")


if __name__ == "__main__":
    main()
