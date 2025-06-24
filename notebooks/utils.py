import pandas as pd

def analyze_column(df, col, top_n=10):
    """
    Analyze and print detailed statistics for a specific column in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    col : str
        The column name to analyze
    top_n : int, default=10
        Number of top values to display
    
    Returns:
    --------
    dict : Dictionary containing the analysis results for programmatic use
    """
    
    if col not in df.columns:
        print(f"❌ Error: Column '{col}' not found in DataFrame")
        return None
    
    print("="*70)
    print(f"VALUE COUNTS FOR '{col.upper()}' COLUMN")
    print("="*70)
    
    # Get value counts and calculate percentages
    value_counts = df[col].value_counts()
    total_non_null = df[col].count()
    total_records = len(df)
    null_count = df[col].isnull().sum()
    null_percentage = (null_count / total_records) * 100
    
    print(f"Non-null records: {total_non_null:,} out of {total_records:,} total records")
    print(f"Null percentage: {null_percentage:.2f}%\n")
    
    # Show top N values with counts and percentages
    print(f"Top {min(top_n, len(value_counts))} values:")
    for i, (value, count) in enumerate(value_counts.head(top_n).items()):
        percentage = (count / total_non_null) * 100
        print(f"  {i+1:2d}. {str(value):<20} | {count:>6,} ({percentage:>5.1f}%)")
    
    if len(value_counts) > top_n:
        print(f"  ... and {len(value_counts) - top_n} more unique values")
    
    print(f"\nTotal unique values: {len(value_counts)}")
    print("\n")
    
    # Return results for programmatic use
    return {
        'column': col,
        'total_records': total_records,
        'non_null_count': total_non_null,
        'null_count': null_count,
        'null_percentage': null_percentage,
        'unique_values': len(value_counts),
        'value_counts': value_counts,
        'top_values': value_counts.head(top_n)
    }


def analyze_categorical_distribution(df, col, top_n=30, frequency_thresholds=[1, 10, 100, 1000], related_column=None):
    """
    Analyze the distribution of a high-cardinality categorical column.
    Useful for columns like 'model', 'manufacturer', etc.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    col : str
        The column name to analyze
    top_n : int, default=30
        Number of top values to display
    frequency_thresholds : list, default=[1, 10, 100, 1000]
        Frequency thresholds to analyze distribution
    related_column : str, optional
        Additional column to display alongside the main column (e.g., 'manufacturer' when analyzing 'model')
    
    Returns:
    --------
    dict : Dictionary containing the analysis results
    """
    
    if col not in df.columns:
        print(f"❌ Error: Column '{col}' not found in DataFrame")
        return None
    
    if related_column and related_column not in df.columns:
        print(f"⚠️  Warning: Related column '{related_column}' not found in DataFrame. Ignoring.")
        related_column = None
    
    print("="*60)
    print(f"{col.upper()} COLUMN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Basic statistics
    unique_values = df[col].nunique()
    total_records = len(df)
    value_counts = df[col].value_counts()
    
    print(f"Total records in dataset: {total_records:,}")
    print(f"Number of distinct {col}s: {unique_values:,}")
    print(f"Average records per {col}: {total_records/unique_values:.1f}")
    
    # Show top N most common values
    if related_column:
        print(f"\nTop {min(top_n, len(value_counts))} most common {col}s (with {related_column}):")
        
        # Create a mapping of main column to most common related column value
        related_mapping = df.groupby(col)[related_column].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown').to_dict()
        
        for i, (value, count) in enumerate(value_counts.head(top_n).items()):
            percentage = (count / total_records) * 100
            
            # Special case: if main column value is 'missing', show 'N/A' for related column
            if str(value).lower() == 'missing':
                related_value = 'N/A'
            else:
                related_value = related_mapping.get(value, 'Unknown')
            
            print(f"  {i+1:2d}. {str(value):<25} | {str(related_value):<15} | {count:>5,} ({percentage:>4.1f}%)")
    else:
        print(f"\nTop {min(top_n, len(value_counts))} most common {col}s:")
        for i, (value, count) in enumerate(value_counts.head(top_n).items()):
            percentage = (count / total_records) * 100
            print(f"  {i+1:2d}. {str(value):<30} | {count:>5,} ({percentage:>4.1f}%)")
    
    # Calculate coverage of top N values
    top_n_actual = min(top_n, len(value_counts))
    top_n_coverage_count = value_counts.head(top_n_actual).sum()
    top_n_coverage_percentage = (top_n_coverage_count / total_records) * 100
    
    print(f"\nTop {top_n_actual} {col}s cover {top_n_coverage_count:,} records ({top_n_coverage_percentage:.1f}% of dataset)")
    
    # Distribution analysis by frequency thresholds
    print(f"\nDistribution by frequency thresholds:")
    distribution_stats = {}
    
    for threshold in sorted(frequency_thresholds):
        if threshold == 1:
            count = (value_counts == threshold).sum()
            print(f"{col.capitalize()}s appearing exactly {threshold} time: {count:,}")
        else:
            count = (value_counts >= threshold).sum()
            print(f"{col.capitalize()}s appearing {threshold}+ times: {count:,}")
        distribution_stats[f'{threshold}+' if threshold > 1 else 'exactly_1'] = count
    
    # Additional insights
    median_frequency = value_counts.median()
    mean_frequency = value_counts.mean()
    
    return {
        'column': col,
        'total_records': total_records,
        'unique_values': unique_values,
        'avg_records_per_value': total_records/unique_values,
        'value_counts': value_counts,
        'top_values': value_counts.head(top_n),
        'distribution_stats': distribution_stats,
        'median_frequency': median_frequency,
        'mean_frequency': mean_frequency,
        'most_frequent': {
            'value': value_counts.index[0],
            'count': value_counts.iloc[0]
        },
        'related_column': related_column
    }

def simplify_high_cardinality_column(df, target_column, related_column, top_n=40, print_stats=True):
    """
    Simplify a high-cardinality column by keeping the top N most frequent values
    and converting all others to "other_<related_column_value>" format.
    
    This is useful for columns like 'model' (with 'manufacturer') or 'region' (with 'state')
    where there are too many unique values for effective one-hot encoding.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    target_column : str
        The column name to simplify (e.g., 'model', 'region')
    related_column : str
        The related column to use for grouping (e.g., 'manufacturer', 'state')
    top_n : int, default=40
        Number of top values to keep unchanged
    print_stats : bool, default=True
        Whether to print statistics about the transformation
    
    Returns:
    --------
    pandas.Series : The simplified column values
    
    Example:
    --------
    # Simplify model column
    df['model_simplified'] = simplify_high_cardinality_column(
        df, 'model', 'manufacturer', top_n=40
    )
    
    # Simplify region column  
    df['region_simplified'] = simplify_high_cardinality_column(
        df, 'region', 'state', top_n=50
    )
    """
    
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")
    
    if related_column not in df.columns:
        raise ValueError(f"Related column '{related_column}' not found in DataFrame")
    
    if print_stats:
        print("="*60)
        print(f"SIMPLIFYING '{target_column.upper()}' COLUMN (TOP {top_n})")
        print("="*60)
    
    # Get the top N values based on frequency
    top_n_values = df[target_column].value_counts().head(top_n).index.tolist()
    
    if print_stats:
        print(f"Top {top_n} {target_column}s identified: {len(top_n_values)} values")
        if len(top_n_values) <= 10:
            print(f"Top {len(top_n_values)} {target_column}s:", top_n_values)
        else:
            print(f"Top 10 {target_column}s:", top_n_values[:10], f"... (and {len(top_n_values)-10} more)")
    
    # Create a function to process the values
    def process_value(row):
        target_value = row[target_column]
        related_value = row[related_column]
        
        if target_value in top_n_values:
            return target_value
        else:
            return f"other_{related_value}"
    
    # Apply the processing
    simplified_column = df.apply(process_value, axis=1)
    
    if print_stats:
        # Show the results
        original_unique = df[target_column].nunique()
        simplified_unique = simplified_column.nunique()
        
        print(f"\nOriginal unique {target_column}s: {original_unique:,}")
        print(f"Simplified unique {target_column}s: {simplified_unique:,}")
        print(f"Reduction: {original_unique - simplified_unique:,} values ({((original_unique - simplified_unique)/original_unique)*100:.1f}%)")
        
        print(f"\nNew {target_column} distribution (top 15):")
        value_counts = simplified_column.value_counts().head(15)
        for i, (value, count) in enumerate(value_counts.items(), 1):
            percentage = (count / len(df)) * 100
            print(f"  {i:2d}. {value:<30} | {count:>6,} ({percentage:4.1f}%)")
    
    return simplified_column


