"""
Data Preprocessing Module for ACIS Insurance Analytics
Handles data loading, cleaning, transformation, and feature engineering
"""

from datetime import datetime
import os
import warnings
import pandas as pd
import numpy as np


warnings.filterwarnings('ignore')


class InsuranceDataPreprocessor:
    """
    Class for preprocessing insurance claim data
    """

    def __init__(self, file_path=None):
        """
        Initialize preprocessor with file path

        Args:
            file_path (str): Path to data file
        """
        self.file_path = file_path
        self.df = None
        self.raw_df = None
        self.metadata = {}
        self.categorical_cols = []
        self.numerical_cols = []
        self.quality_report = {}
        self.outlier_report = {}

    def load_data(self, file_path=None, delimiter=None, encoding='utf-8'):
        """
        Load data from specified file path with intelligent detection

        Args:
            file_path (str): Path to data file
            delimiter (str): Delimiter for text file (auto-detected if None)
            encoding (str): File encoding

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if file_path:
            self.file_path = file_path

        # Use default path if not specified
        if self.file_path is None:
            self.file_path = 'data/raw/MachineLearningRating_v3.txt'

        print(f"Loading data from: {self.file_path}")

        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                print(f"File not found: {self.file_path}")
                # Try alternative paths
                possible_paths = [
                    self.file_path,
                    os.path.join('..', self.file_path),
                    os.path.join('../..', self.file_path),
                    os.path.join(
                        'data', 'raw', 'MachineLearningRating_v3.txt'),
                    os.path.join('..', 'data', 'raw',
                                 'MachineLearningRating_v3.txt'),
                    os.path.join('../..', 'data', 'raw',
                                 'MachineLearningRating_v3.txt'),
                    'MachineLearningRating_v3.txt'
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        self.file_path = path
                        print(f"Found file at: {path}")
                        break
                else:
                    raise FileNotFoundError(
                        print("Data file not found in any expected location"))

            # Get file size
            file_size = os.path.getsize(self.file_path)
            print(f"File size: {file_size/1024/1024:.2f} MB")

            # Try to detect encoding by trying common encodings
            if encoding is None:
                print("Trying to detect file encoding...")
                possible_encodings = ['utf-8', 'latin-1',
                                      'iso-8859-1', 'cp1252', 'utf-16']
                for enc in possible_encodings:
                    try:
                        with open(self.file_path, 'r', encoding=enc) as f:
                            f.read(1024)  # Read first 1KB to test encoding
                        encoding = enc
                        print(f"Encoding detected: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    encoding = 'utf-8'
                    print("Could not detect encoding, using utf-8 as default")

            # Determine file type and load accordingly
            file_extension = os.path.splitext(self.file_path)[1].lower()

            if file_extension == '.txt':
                self.raw_df = self._load_text_file(delimiter, encoding)
            elif file_extension == '.csv':
                self.raw_df = pd.read_csv(
                    self.file_path, encoding=encoding, low_memory=False)
                print("✓ Loaded CSV file")
            elif file_extension in ['.xlsx', '.xls']:
                self.raw_df = pd.read_excel(self.file_path)
                print("✓ Loaded Excel file")
            else:
                # Try to load as text file with auto-detection
                self.raw_df = self._load_text_file(delimiter, encoding)

            self.df = self.raw_df.copy()
            self.metadata['original_shape'] = self.df.shape
            self.metadata['file_path'] = self.file_path
            self.metadata['file_size_mb'] = file_size / (1024 * 1024)
            self.metadata['encoding'] = encoding

            print(
                f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

            return self.df

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _load_text_file(self, delimiter=None, encoding='utf-8'):
        """
        Load text file with intelligent delimiter detection

        Args:
            delimiter (str): Delimiter for text file
            encoding (str): File encoding

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print("Loading text file with intelligent parsing...")

        # First, inspect the file structure
        try:
            with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                first_lines = [next(f) for _ in range(10) if f]
        except UnicodeDecodeError:
            # Try with latin-1 if utf-8 fails
            print("UTF-8 decoding failed, trying latin-1...")
            encoding = 'latin-1'
            with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                first_lines = [next(f) for _ in range(10) if f]

        print("First few lines of the file (first 100 chars each):")
        for i, line in enumerate(first_lines[:3]):
            print(f"Line {i+1}: {line[:100].rstrip()}...")

        # Try to detect delimiter if not provided
        if delimiter is None:
            delimiter = self._detect_delimiter(first_lines)
            print(
                f"Auto-detected delimiter: '{delimiter}' (shown as \\t if tab)")

        # Try different parsing strategies
        try:
            # First attempt: Standard read with detected delimiter
            df = pd.read_csv(
                self.file_path,
                delimiter=delimiter,
                encoding=encoding,
                low_memory=False,
                on_bad_lines='warn'
            )
            print(f"✓ Successfully loaded with delimiter '{delimiter}'")
        except Exception as e:
            print(f"First attempt failed: {str(e)}")

            # Second attempt: Try with Python engine
            try:
                df = pd.read_csv(
                    self.file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    engine='python',
                    on_bad_lines='warn'
                )
                print("Successfully loaded with Python engine")
            except Exception as e2:
                print(f"Second attempt failed: {str(e2)}")

                # Third attempt: Try reading with no header first to inspect
                print("Trying to read without header to inspect structure...")
                try:
                    df_no_header = pd.read_csv(
                        self.file_path,
                        delimiter=delimiter,
                        encoding=encoding,
                        header=None,
                        nrows=100,
                        engine='python'
                    )
                    print(f"Data shape without header: {df_no_header.shape}")
                    print("First few rows without header:")
                    print(df_no_header.head())

                    # Ask user for number of header rows
                    header_rows = 1  # Default assumption
                    if len(df_no_header) > 0:
                        # Try to find header by looking for column name patterns
                        potential_headers = 0
                        for i in range(min(5, len(df_no_header))):
                            row_str = ' '.join(str(x)
                                               for x in df_no_header.iloc[i].values)
                            if any(keyword in row_str.lower() for keyword in
                                   ['id', 'date', 'name', 'type', 'total', 'premium']):
                                potential_headers += 1

                        if potential_headers > 0:
                            header_rows = potential_headers
                            print(
                                f"Detected {header_rows} potential header rows")

                    # Load with appropriate header
                    df = pd.read_csv(
                        self.file_path,
                        delimiter=delimiter,
                        encoding=encoding,
                        header=list(range(header_rows)),
                        engine='python',
                        on_bad_lines='skip'
                    )
                    print(
                        f"✓ Successfully loaded with {header_rows} header rows")

                except Exception as e3:
                    print(f"All attempts failed: {str(e3)}")
                    raise

        return df

    def _detect_delimiter(self, sample_lines):
        """
        Detect delimiter from sample lines

        Args:
            sample_lines (list): Sample lines from file

        Returns:
            str: Detected delimiter
        """
        possible_delimiters = ['\t', ',', ';', '|', ' ']
        delimiter_counts = {}

        for delim in possible_delimiters:
            delimiter_counts[delim] = sum(
                line.count(delim) for line in sample_lines)

        # Find delimiter with maximum and consistent count
        for delim in ['\t', ',', ';', '|']:  # Check common delimiters first
            if delimiter_counts[delim] > 0:
                counts = [line.count(delim) for line in sample_lines]
                if all(count == counts[0] for count in counts) and counts[0] > 0:
                    print(
                        f"Consistent delimiter found: '{delim}' with {counts[0]} columns")
                    return delim

        # If no consistent delimiter found, use the most common one
        detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        print(
            f"Using most common delimiter: '{detected_delimiter}' with {delimiter_counts[detected_delimiter]} total occurrences")

        return detected_delimiter

    def inspect_data_structure(self):
        """
        Inspect and print data structure information
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return

        print("=" * 80)
        print("DATA STRUCTURE INSPECTION")
        print("=" * 80)

        print(
            f"\nDataset shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nColumn names and data types:")
        dtype_summary = self.df.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"  {dtype}: {count} columns")

        print("\nDetailed column information:")
        for i, (col, dtype) in enumerate(self.df.dtypes.items()):
            unique_count = self.df[col].nunique(
            ) if col in self.df.columns else 0
            missing_count = self.df[col].isnull(
            ).sum() if col in self.df.columns else 0
            print(f"{i+1:3}. {col:<30} : {dtype:<15} | Unique: {unique_count:>5} | Missing: {missing_count:>5} ({missing_count/len(self.df)*100:.1f}%)")

        print(
            f"\nMemory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Check for multi-index columns
        if isinstance(self.df.columns, pd.MultiIndex):
            print("\n⚠ Multi-level columns detected:")
            for i, level in enumerate(self.df.columns.levels):
                print(f"  Level {i}: {len(level)} unique values")

    def validate_data_structure(self):
        """
        Validate and convert data types
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return

        print("=" * 80)
        print("DATA TYPE VALIDATION")
        print("=" * 80)

        # Flatten multi-index columns if they exist
        if isinstance(self.df.columns, pd.MultiIndex):
            print("Flattening multi-index columns...")
            self.df.columns = [
                '_'.join(filter(None, map(str, col))).strip() for col in self.df.columns]
            print(f"New column names: {list(self.df.columns[:10])}...")

        # Standardize column names
        print("\nStandardizing column names...")
        original_columns = self.df.columns.tolist()
        self.df.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('.', '_').lower()
                           for col in self.df.columns]

        # Create mapping of original to standardized names
        column_mapping = dict(zip(original_columns, self.df.columns))
        self.metadata['column_mapping'] = column_mapping

        print(f"Standardized {len(column_mapping)} column names")
        print("Sample mapping:")
        for orig, new in list(column_mapping.items())[:10]:
            print(f"  {orig} → {new}")

        # Date columns conversion
        date_patterns = {
            'transactionmonth': ['transactionmonth', 'transaction_month', 'trans_month', 'month', 'date'],
            'vehicleintrodate': ['vehicleintrodate', 'vehicle_intro_date', 'intro_date', 'vehicledate'],
            'registrationyear': ['registrationyear', 'reg_year', 'registration_year']
        }

        print("\nDate column conversion:")
        for target_col, possible_names in date_patterns.items():
            found_col = None
            for name in possible_names:
                if name in self.df.columns:
                    found_col = name
                    break

            if found_col:
                try:
                    if target_col == 'registrationyear':
                        # Handle registration year as integer or date
                        self.df[target_col] = pd.to_numeric(
                            self.df[found_col], errors='coerce').fillna(0).astype(int)
                        print(
                            f"✓ {found_col} → {target_col}: Converted to integer year")
                    else:
                        self.df[target_col] = pd.to_datetime(
                            self.df[found_col], errors='coerce')
                        print(
                            f"✓ {found_col} → {target_col}: Converted to datetime")
                except Exception as e:
                    print(f"✗ {found_col}: Could not convert - {str(e)}")
            else:
                # Try to find columns with date-like names
                date_like_cols = [col for col in self.df.columns if any(
                    term in col.lower() for term in ['date', 'month', 'year', 'time'])]
                # Check first 3 date-like columns
                for col in date_like_cols[:3]:
                    try:
                        sample = self.df[col].dropna(
                        ).iloc[0] if not self.df[col].dropna().empty else None
                        if sample and (isinstance(sample, str) and any(char.isdigit() for char in str(sample))):
                            self.df[f"{col}_as_date"] = pd.to_datetime(
                                self.df[col], errors='coerce')
                            print(
                                f"✓ Found date-like column: {col} (added as {col}_as_date)")
                    except:
                        pass

        # Categorical columns identification
        self.categorical_cols = self.df.select_dtypes(
            include=['object']).columns.tolist()
        self.numerical_cols = self.df.select_dtypes(
            include=[np.number]).columns.tolist()

        print(f"\n✓ Categorical columns: {len(self.categorical_cols)}")
        print(f"✓ Numerical columns: {len(self.numerical_cols)}")

        self.metadata['categorical_cols'] = self.categorical_cols
        self.metadata['numerical_cols'] = self.numerical_cols

        # Check for specific required columns
        required_columns = ['totalpremium', 'totalclaims',
                            'suminsured', 'province', 'vehicletype']
        missing_required = [
            col for col in required_columns if col not in self.df.columns]

        if missing_required:
            print(f"\n⚠ Warning: Missing required columns: {missing_required}")

            # Try to find similar column names
            print("Looking for similar column names...")
            all_columns = self.df.columns.tolist()
            for missing_col in missing_required:
                similar = [col for col in all_columns if missing_col in col.lower(
                ) or col.lower() in missing_col]
                if similar:
                    print(f"  {missing_col} might be: {similar}")

                    # Create aliases for similar columns
                    for sim_col in similar:
                        if sim_col not in self.df.columns:
                            continue
                        alias_name = missing_col
                        self.df[alias_name] = self.df[sim_col]
                        print(f"    Created alias: {sim_col} → {alias_name}")
        else:
            print("\n✓ All required columns found or created")

        print("\n✓ Data structure validation complete")

    def assess_data_quality(self, save_report=True):
        """
        Assess data quality including missing values and basic statistics

        Args:
            save_report (bool): Whether to save quality report to file

        Returns:
            dict: Quality assessment report
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None

        print("=" * 80)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 80)

        quality_report = {}

        # 1. Missing values analysis
        print("\n1. Missing Values Analysis:")
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100

        quality_report['missing_values'] = pd.DataFrame({
            'missing_count': missing_data,
            'missing_percentage': missing_percentage
        }).sort_values('missing_percentage', ascending=False)

        # Display top columns with missing values
        top_missing = quality_report['missing_values'].head(15)
        print("Top 15 columns with missing values:")
        for idx, row in top_missing.iterrows():
            print(
                f"  {idx:<35}: {row['missing_count']:>8} ({row['missing_percentage']:>6.2f}%)")

        total_missing = missing_data.sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_overall_pct = (total_missing / total_cells) * 100
        print(
            f"\nOverall missing values: {total_missing:,} / {total_cells:,} ({missing_overall_pct:.2f}%)")

        # 2. Duplicate analysis
        print("\n2. Duplicate Analysis:")
        duplicate_rows = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(self.df)) * 100

        quality_report['duplicate_rows'] = duplicate_rows
        quality_report['duplicate_percentage'] = duplicate_percentage

        print(
            f"Exact duplicate rows: {duplicate_rows:,} ({duplicate_percentage:.2f}%)")

        # Check for near duplicates based on key columns
        key_columns = [col for col in ['underwrittencoverid', 'policyid', 'totalpremium', 'totalclaims']
                       if col in self.df.columns]
        if len(key_columns) >= 2:
            near_duplicates = self.df.duplicated(
                subset=key_columns, keep=False).sum()
            near_dup_percentage = (near_duplicates / len(self.df)) * 100
            quality_report['near_duplicates'] = near_duplicates
            quality_report['near_duplicate_percentage'] = near_dup_percentage
            print(
                f"Near duplicates (based on key columns): {near_duplicates:,} ({near_dup_percentage:.2f}%)")

        # 3. Unique values analysis
        print("\n3. Unique Values Analysis:")
        unique_counts = {}

        # For categorical columns
        print("Categorical columns (unique values):")
        # Limit to first 15 for performance
        for col in self.categorical_cols[:15]:
            unique_counts[col] = self.df[col].nunique()
            # Only show columns with reasonable number of categories
            if unique_counts[col] < 50:
                print(f"  {col:<35}: {unique_counts[col]:>6} unique values")

        # For numerical columns, check cardinality
        print("\nNumerical columns (cardinality):")
        for col in self.numerical_cols[:15]:
            unique_pct = (self.df[col].nunique() / len(self.df)) * 100
            if unique_pct < 10:  # Low cardinality numerical columns
                print(
                    f"  {col:<35}: {self.df[col].nunique():>6} unique ({unique_pct:.1f}%)")

        quality_report['unique_counts'] = unique_counts

        # 4. Basic statistics for numerical columns
        print("\n4. Basic Statistics for Key Columns:")
        financial_cols = ['totalpremium', 'totalclaims',
                          'suminsured', 'calculatedpremiumperterm']
        available_financial = [
            col for col in financial_cols if col in self.df.columns]

        if available_financial:
            stats_df = self.df[available_financial].describe().T
            stats_df['zeros'] = [(self.df[col] == 0).sum()
                                 for col in available_financial]
            stats_df['zeros_pct'] = stats_df['zeros'] / len(self.df) * 100

            print("\nFinancial Columns Summary:")
            display_cols = ['count', 'mean', 'std', 'min',
                            '25%', '50%', '75%', 'max', 'zeros', 'zeros_pct']
            display_stats = stats_df[display_cols].copy()
            display_stats.columns = [
                'Count', 'Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Zeros', 'Zeros%']

            for idx, row in display_stats.iterrows():
                print(f"\n{idx}:")
                for col in display_stats.columns:
                    if col in ['Zeros%']:
                        print(f"  {col:<8}: {row[col]:>10.2f}%")
                    elif col in ['Mean', 'Std']:
                        print(f"  {col:<8}: {row[col]:>10,.2f}")
                    else:
                        print(f"  {col:<8}: {row[col]:>10,}")

            quality_report['financial_stats'] = stats_df

        # 5. Data type distribution
        print("\n5. Data Type Distribution:")
        dtype_dist = self.df.dtypes.value_counts()
        for dtype, count in dtype_dist.items():
            print(f"  {str(dtype):<20}: {count:>5} columns")

        quality_report['dtype_distribution'] = dtype_dist

        # 6. Memory usage
        memory_by_column = self.df.memory_usage(deep=True)
        total_memory_mb = memory_by_column.sum() / 1024**2

        print(f"\n6. Memory Usage:")
        print(f"  Total memory: {total_memory_mb:.2f} MB")
        print(
            f"  Average per column: {total_memory_mb / len(self.df.columns):.2f} MB")

        # Top memory consuming columns
        top_memory = memory_by_column.nlargest(10) / 1024**2
        print("\n  Top 10 memory-consuming columns:")
        for col, mem in top_memory.items():
            print(f"    {col:<35}: {mem:>8.2f} MB")

        quality_report['memory_usage'] = {
            'total_mb': total_memory_mb,
            'by_column_mb': (memory_by_column / 1024**2).to_dict()
        }

        # 7. Date range analysis
        print("\n7. Date Range Analysis:")
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            for col in date_cols:
                min_date = self.df[col].min()
                max_date = self.df[col].max()
                date_range = (
                    max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else None

                print(
                    f"  {col:<20}: {min_date} to {max_date} ({date_range} days)")

                quality_report['date_ranges'] = quality_report.get(
                    'date_ranges', {})
                quality_report['date_ranges'][col] = {
                    'min': min_date,
                    'max': max_date,
                    'range_days': date_range
                }
        else:
            print("  No datetime columns found")

        # 8. Basic dataset statistics
        quality_report['basic_stats'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'total_cells': total_cells,
            'total_missing': total_missing,
            'missing_percentage': missing_overall_pct,
            'memory_usage_mb': total_memory_mb,
        }

        # Calculate data quality score
        completeness_score = 100 - missing_overall_pct
        uniqueness_score = 100 - duplicate_percentage
        quality_score = (completeness_score * 0.7) + (uniqueness_score * 0.3)

        quality_report['quality_score'] = {
            'completeness': completeness_score,
            'uniqueness': uniqueness_score,
            'overall': quality_score
        }

        print(f"\n8. Data Quality Score: {quality_score:.1f}/100")
        print(f"   - Completeness: {completeness_score:.1f}/100")
        print(f"   - Uniqueness: {uniqueness_score:.1f}/100")

        self.quality_report = quality_report

        # Save report if requested
        if save_report:
            self._save_quality_report(quality_report)

        return quality_report

    def _save_quality_report(self, quality_report):
        """Save quality report to CSV files"""

        os.makedirs('../data/outputs', exist_ok=True)

        # Save missing values summary
        if 'missing_values' in quality_report:
            quality_report['missing_values'].to_csv(
                '../data/outputs/missing_values_summary.csv')
            print(
                "✓ Saved missing values summary to data/outputs/missing_values_summary.csv")

        # Save basic stats
        if 'basic_stats' in quality_report:
            pd.Series(quality_report['basic_stats']).to_csv(
                '../data/outputs/basic_stats.csv')
            print("✓ Saved basic statistics to data/outputs/basic_stats.csv")

        # Save financial stats
        if 'financial_stats' in quality_report:
            quality_report['financial_stats'].to_csv(
                '../data/outputs/financial_stats.csv')
            print("✓ Saved financial statistics to data/outputs/financial_stats.csv")

    def handle_missing_values(self, strategy='median', categorical_strategy='unknown'):
        """
        Handle missing values based on specified strategy

        Args:
            strategy (str): Strategy for numerical columns ('mean', 'median', 'mode', 'drop', 'interpolate')
            categorical_strategy (str): Strategy for categorical columns ('mode', 'drop', 'unknown', 'missing')
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return

        print("=" * 80)
        print("HANDLING MISSING VALUES")
        print("=" * 80)

        initial_missing = self.df.isnull().sum().sum()
        print(f"Total missing values before handling: {initial_missing:,}")

        # Handle numerical columns
        num_cols_with_missing = [
            col for col in self.numerical_cols if self.df[col].isnull().any()]

        if num_cols_with_missing:
            print(
                f"\nHandling missing values in {len(num_cols_with_missing)} numerical columns...")

            for col in num_cols_with_missing:
                missing_count = self.df[col].isnull().sum()
                missing_pct = (missing_count / len(self.df)) * 100

                if missing_pct > 50:
                    print(
                        f"  ⚠ {col:<35}: {missing_count:>6} missing ({missing_pct:>5.1f}%) - Too many missing, considering dropping")
                    if strategy == 'drop':
                        self.df.dropna(subset=[col], inplace=True)
                        print(f"    Dropped rows with missing {col}")
                    else:
                        # For columns with >50% missing, use a conservative approach
                        if self.df[col].dtype in [np.int64, np.int32]:
                            self.df[col].fillna(0, inplace=True)
                            print(f"    Filled with 0 (integer column)")
                        else:
                            self.df[col].fillna(
                                self.df[col].median(), inplace=True)
                            print(f"    Filled with median")
                else:
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                        self.df[col].fillna(fill_value, inplace=True)
                        print(
                            f"  ✓ {col:<35}: Filled {missing_count:>6} values with mean ({fill_value:.2f})")
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                        self.df[col].fillna(fill_value, inplace=True)
                        print(
                            f"  ✓ {col:<35}: Filled {missing_count:>6} values with median ({fill_value:.2f})")
                    elif strategy == 'mode':
                        fill_value = self.df[col].mode(
                        )[0] if not self.df[col].mode().empty else 0
                        self.df[col].fillna(fill_value, inplace=True)
                        print(
                            f"  ✓ {col:<35}: Filled {missing_count:>6} values with mode ({fill_value})")
                    elif strategy == 'interpolate':
                        self.df[col] = self.df[col].interpolate(
                            method='linear', limit_direction='both')
                        print(
                            f"  ✓ {col:<35}: Interpolated {missing_count:>6} values")
                    elif strategy == 'drop':
                        self.df.dropna(subset=[col], inplace=True)
                        print(
                            f"  ✓ {col:<35}: Dropped rows with missing values")

        # Handle categorical columns
        cat_cols_with_missing = [
            col for col in self.categorical_cols if self.df[col].isnull().any()]

        if cat_cols_with_missing:
            print(
                f"\nHandling missing values in {len(cat_cols_with_missing)} categorical columns...")

            for col in cat_cols_with_missing:
                missing_count = self.df[col].isnull().sum()
                missing_pct = (missing_count / len(self.df)) * 100

                if missing_pct > 30:
                    print(
                        f"  ⚠ {col:<35}: {missing_count:>6} missing ({missing_pct:>5.1f}%) - High missing percentage")

                if categorical_strategy == 'mode':
                    if not self.df[col].mode().empty:
                        fill_value = self.df[col].mode()[0]
                        self.df[col].fillna(fill_value, inplace=True)
                        print(
                            f"  ✓ {col:<35}: Filled {missing_count:>6} values with mode ('{fill_value}')")
                    else:
                        self.df[col].fillna('Unknown', inplace=True)
                        print(
                            f"  ✓ {col:<35}: Filled {missing_count:>6} values with 'Unknown'")
                elif categorical_strategy == 'unknown':
                    self.df[col].fillna('Unknown', inplace=True)
                    print(
                        f"  ✓ {col:<35}: Filled {missing_count:>6} values with 'Unknown'")
                elif categorical_strategy == 'missing':
                    self.df[col].fillna('Missing', inplace=True)
                    print(
                        f"  ✓ {col:<35}: Filled {missing_count:>6} values with 'Missing'")
                elif categorical_strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                    print(f"  ✓ {col:<35}: Dropped rows with missing values")

        final_missing = self.df.isnull().sum().sum()
        reduction = initial_missing - final_missing

        print(f"\n{'='*50}")
        print(f"Missing values handled:")
        print(f"  Before: {initial_missing:,}")
        print(f"  After:  {final_missing:,}")
        print(
            f"  Reduction: {reduction:,} ({reduction/initial_missing*100:.1f}%)")
        print(f"{'='*50}")

        # Update metadata
        self.metadata['missing_values_handled'] = {
            'initial': initial_missing,
            'final': final_missing,
            'reduction': reduction,
            'strategy': strategy,
            'categorical_strategy': categorical_strategy
        }

    def engineer_features(self):
        """
        Create new features for analysis
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return

        print("=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)

        original_cols = set(self.df.columns)
        new_features = []

        # 1. Calculate Loss Ratio
        if all(col in self.df.columns for col in ['totalclaims', 'totalpremium']):
            self.df['lossratio'] = self.df['totalclaims'] / \
                self.df['totalpremium'].replace(0, np.nan)
            self.df['lossratio'].fillna(0, inplace=True)
            new_features.append('lossratio')
            print("✓ Created feature: lossratio (TotalClaims / TotalPremium)")

            # Binary claim indicator
            self.df['has_claim'] = (self.df['totalclaims'] > 0).astype(int)
            new_features.append('has_claim')
            print("✓ Created feature: has_claim (1 if claim > 0)")

            # Claim severity
            self.df['claim_severity'] = self.df['totalclaims'] / \
                self.df['totalpremium'].replace(0, np.nan)
            self.df['claim_severity'].fillna(0, inplace=True)
            new_features.append('claim_severity')
            print("✓ Created feature: claim_severity")

        # 2. Vehicle-related features
        if 'registrationyear' in self.df.columns:
            current_year = datetime.now().year
            self.df['vehicle_age'] = current_year - self.df['registrationyear']
            self.df['vehicle_age'] = self.df['vehicle_age'].clip(
                lower=0, upper=50)  # Clip unrealistic values
            new_features.append('vehicle_age')
            print("✓ Created feature: vehicle_age")

            # Age categories
            bins = [0, 3, 7, 12, 20, 100]
            labels = ['New (0-3)', 'Young (4-7)', 'Mid (8-12)',
                      'Old (13-20)', 'Vintage (20+)']
            self.df['vehicle_age_category'] = pd.cut(
                self.df['vehicle_age'], bins=bins, labels=labels, right=False)
            new_features.append('vehicle_age_category')
            print("✓ Created feature: vehicle_age_category")

        # 3. Premium-related features
        if all(col in self.df.columns for col in ['totalpremium', 'customvalueestimate']):
            self.df['premium_to_value_ratio'] = self.df['totalpremium'] / \
                self.df['customvalueestimate'].replace(0, np.nan)
            self.df['premium_to_value_ratio'].fillna(0, inplace=True)
            new_features.append('premium_to_value_ratio')
            print("✓ Created feature: premium_to_value_ratio")

        if all(col in self.df.columns for col in ['calculatedpremiumperterm', 'totalpremium']):
            self.df['premium_variance'] = self.df['totalpremium'] - \
                self.df['calculatedpremiumperterm']
            new_features.append('premium_variance')
            print("✓ Created feature: premium_variance")

        # 4. Temporal features
        if 'transactionmonth' in self.df.columns:
            self.df['transaction_yearmonth'] = self.df['transactionmonth'].dt.to_period(
                'M').astype(str)
            self.df['transaction_month'] = self.df['transactionmonth'].dt.month
            self.df['transaction_quarter'] = self.df['transactionmonth'].dt.quarter
            self.df['transaction_year'] = self.df['transactionmonth'].dt.year
            self.df['transaction_dayofweek'] = self.df['transactionmonth'].dt.dayofweek
            self.df['is_weekend'] = self.df['transaction_dayofweek'].isin([
                                                                          5, 6]).astype(int)

            new_features.extend(['transaction_yearmonth', 'transaction_month', 'transaction_quarter',
                                 'transaction_year', 'transaction_dayofweek', 'is_weekend'])
            print("✓ Created time-based features")

        # 5. Risk segmentation features
        if 'lossratio' in self.df.columns:
            # Create risk categories based on loss ratio
            risk_bins = [-np.inf, 0.1, 0.3, 0.6, 1.0, np.inf]
            risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            self.df['risk_category'] = pd.cut(
                self.df['lossratio'], bins=risk_bins, labels=risk_labels)
            new_features.append('risk_category')
            print("✓ Created feature: risk_category")

        # 6. Customer-related features
        if 'numberofvehiclesinfleet' in self.df.columns:
            self.df['is_fleet'] = (
                self.df['numberofvehiclesinfleet'] > 1).astype(int)
            new_features.append('is_fleet')
            print("✓ Created feature: is_fleet")

        # 7. Interaction features
        if all(col in self.df.columns for col in ['gender', 'maritalstatus']):
            self.df['gender_marital'] = self.df['gender'] + \
                '_' + self.df['maritalstatus']
            new_features.append('gender_marital')
            print("✓ Created feature: gender_marital (interaction)")

        if all(col in self.df.columns for col in ['province', 'vehicletype']):
            self.df['province_vehicletype'] = self.df['province'] + \
                '_' + self.df['vehicletype']
            new_features.append('province_vehicletype')
            print("✓ Created feature: province_vehicletype (interaction)")

        # Update metadata
        new_cols = set(self.df.columns) - original_cols
        self.metadata['engineered_features'] = list(new_cols)
        self.metadata['feature_count'] = len(new_features)

        print(f"\n✓ Feature engineering complete:")
        print(f"  Original columns: {len(original_cols)}")
        print(f"  New columns: {len(new_cols)}")
        print(f"  Total columns: {len(self.df.columns)}")
        print(f"\nNew features created: {new_features}")

    def _calculate_zscore(self, data):
        """
        Calculate z-score without using scipy.stats

        Args:
            data (pd.Series): Data series

        Returns:
            np.array: Z-scores
        """
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros(len(data))
        return (data - mean) / std

    def detect_outliers(self, method='iqr', threshold=1.5, columns=None):
        """
        Detect outliers in numerical columns

        Args:
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
            columns (list): Specific columns to analyze (None for all numerical)

        Returns:
            dict: Dictionary with outlier information
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None

        print("=" * 80)
        print("OUTLIER DETECTION")
        print("=" * 80)

        if columns is None:
            columns = self.numerical_cols

        # Filter to columns that exist and have data
        columns = [
            col for col in columns if col in self.df.columns and self.df[col].notna().sum() > 0]

        print(
            f"Analyzing {len(columns)} numerical columns for outliers using {method} method...")

        outlier_report = {}
        total_outliers = 0

        for col in columns:
            try:
                data = self.df[col].dropna()
                if len(data) == 0:
                    continue

                # Initialize variables
                outliers = pd.Series(dtype='bool')
                lower_bound = None
                upper_bound = None

                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR == 0:
                        # Use standard deviation if IQR is zero
                        std = data.std()
                        if std == 0:
                            continue
                        lower_bound = data.mean() - threshold * std
                        upper_bound = data.mean() + threshold * std
                        outliers = self.df[col].apply(
                            lambda x: x < lower_bound or x > upper_bound)
                    else:
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        outliers = (self.df[col] < lower_bound) | (
                            self.df[col] > upper_bound)

                elif method == 'zscore':
                    # Use custom z-score calculation
                    z_scores = np.abs(self._calculate_zscore(data))
                    # Create a boolean mask for outliers in the original dataframe
                    outlier_mask = pd.Series(False, index=self.df.index)
                    outlier_indices = data.index[z_scores > threshold]
                    outlier_mask.loc[outlier_indices] = True
                    outliers = outlier_mask
                else:
                    print(
                        f"  ✗ {col}: Unknown method '{method}'. Use 'iqr' or 'zscore'.")
                    continue

                outlier_count = outliers.sum()
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / len(self.df)) * 100
                    total_outliers += outlier_count

                    outlier_report[col] = {
                        'outlier_count': int(outlier_count),
                        'outlier_percentage': float(outlier_percentage),
                        'min_value': float(data.min()),
                        'max_value': float(data.max()),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'lower_bound': float(lower_bound) if lower_bound is not None else None,
                        'upper_bound': float(upper_bound) if upper_bound is not None else None,
                        # Limit to first 100
                        'outlier_indices': self.df[outliers].index.tolist()[:100]
                    }

                    if outlier_percentage > 5:  # Only show columns with >5% outliers
                        print(f"  ⚠ {col:<35}: {outlier_count:>6} outliers ({outlier_percentage:>5.1f}%) "
                              f"[min: {data.min():.2f}, max: {data.max():.2f}]")

            except Exception as e:
                print(f"  ✗ Error analyzing {col}: {str(e)}")
                continue

        # Summary
        print(f"\nOutlier Detection Summary:")
        print(f"  Total columns analyzed: {len(columns)}")
        print(f"  Columns with outliers: {len(outlier_report)}")
        print(f"  Total outlier instances: {total_outliers:,}")

        if outlier_report:
            # Sort by outlier percentage
            sorted_report = sorted(outlier_report.items(
            ), key=lambda x: x[1]['outlier_percentage'], reverse=True)

            print(f"\nTop 10 columns with highest outlier percentage:")
            for i, (col, stats) in enumerate(sorted_report[:10]):
                print(
                    f"  {i+1:2}. {col:<35}: {stats['outlier_count']:>6} ({stats['outlier_percentage']:>5.1f}%)")

        self.outlier_report = outlier_report

        # Save outlier report
        if outlier_report:
            self._save_outlier_report(outlier_report)

        return outlier_report

    def _save_outlier_report(self, outlier_report):
        """Save outlier report to CSV"""
        import os
        os.makedirs('../data/outputs', exist_ok=True)

        # Convert to DataFrame
        report_df = pd.DataFrame.from_dict(outlier_report, orient='index')
        report_df = report_df.sort_values(
            'outlier_percentage', ascending=False)

        # Save full report
        report_df.to_csv('../data/outputs/outlier_report_full.csv')

        # Save summary
        summary_cols = ['outlier_count', 'outlier_percentage',
                        'min_value', 'max_value', 'mean', 'std']
        existing_cols = [
            col for col in summary_cols if col in report_df.columns]
        if existing_cols:
            summary_df = report_df[existing_cols]
            summary_df.to_csv('../data/outputs/outlier_report_summary.csv')

        print("✓ Saved outlier reports to data/outputs/outlier_report_*.csv")

    def remove_outliers(self, columns=None, method='iqr', threshold=1.5, inplace=True):
        """
        Remove outliers from specified columns

        Args:
            columns (list): Columns to remove outliers from
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
            inplace (bool): Whether to modify the dataframe in place

        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None

        if columns is None:
            columns = list(self.outlier_report.keys()
                           ) if self.outlier_report else self.numerical_cols

        print(f"Removing outliers from {len(columns)} columns...")

        if inplace:
            df_clean = self.df
        else:
            df_clean = self.df.copy()

        initial_rows = len(df_clean)
        outlier_indices = set()

        for col in columns:
            if col not in df_clean.columns:
                continue

            data = df_clean[col].dropna()
            if len(data) == 0:
                continue

            col_outliers = set()  # Initialize as empty set

            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0:
                    continue

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                col_outliers = set(df_clean[(df_clean[col] < lower_bound) | (
                    df_clean[col] > upper_bound)].index)

            elif method == 'zscore':
                # Use custom z-score calculation
                z_scores = np.abs(self._calculate_zscore(data))
                col_outliers = set(
                    data.index[np.where(z_scores > threshold)[0]])

            outlier_indices.update(col_outliers)

        # Remove rows with outliers in any of the specified columns
        rows_removed = len(outlier_indices)
        df_clean = df_clean.drop(index=list(outlier_indices))

        print(f"Outlier removal complete:")
        print(f"  Initial rows: {initial_rows:,}")
        print(f"  Rows removed: {rows_removed:,}")
        print(f"  Final rows: {len(df_clean):,}")
        print(f"  Percentage removed: {rows_removed/initial_rows*100:.2f}%" if initial_rows >
              0 else "  Percentage removed: N/A")

        if inplace:
            self.df = df_clean
            print("✓ Outliers removed in-place")
        else:
            print("✓ Outliers removed from copy")

        return df_clean

    def get_preprocessed_data(self):
        """
        Get the preprocessed dataframe

        Returns:
            pd.DataFrame: Preprocessed data
        """
        return self.df

    def get_metadata(self):
        """
        Get preprocessing metadata

        Returns:
            dict: Metadata dictionary
        """
        return self.metadata

    def save_preprocessed_data(self, output_path):
        """
        Save preprocessed data to file

        Args:
            output_path (str): Path to save preprocessed data
        """
        if self.df is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save to CSV
            self.df.to_csv(output_path, index=False)
            print(f"✓ Preprocessed data saved to: {output_path}")

            # Also save metadata
            metadata_path = output_path.replace('.csv', '_metadata.json')
            import json

            # Convert metadata to JSON serializable format
            metadata_serializable = {}
            for key, value in self.metadata.items():
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    metadata_serializable[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    metadata_serializable[key] = value.tolist()
                elif isinstance(value, np.generic):
                    metadata_serializable[key] = value.item()
                else:
                    metadata_serializable[key] = value

            with open(metadata_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=4, default=str)

            print(f"✓ Metadata saved to: {metadata_path}")
        else:
            print("No data to save. Please preprocess data first.")

    def get_summary(self):
        """
        Get a summary of the preprocessing steps

        Returns:
            dict: Summary dictionary
        """
        summary = {
            'file_path': self.metadata.get('file_path', 'Unknown'),
            'original_shape': self.metadata.get('original_shape', (0, 0)),
            'current_shape': self.df.shape if self.df is not None else (0, 0),
            'categorical_columns': len(self.categorical_cols),
            'numerical_columns': len(self.numerical_cols),
            'quality_score': self.quality_report.get('quality_score', {}).get('overall', 0) if self.quality_report else 0,
            'missing_values_initial': self.metadata.get('missing_values_handled', {}).get('initial', 0),
            'missing_values_final': self.df.isnull().sum().sum() if self.df is not None else 0,
            'engineered_features': len(self.metadata.get('engineered_features', [])),
            'outlier_columns': len(self.outlier_report)
        }

        return summary
