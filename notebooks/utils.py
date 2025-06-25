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


# =============================================================================
# FEATURE ENGINEERING HELPERS FOR MODELING PIPELINE
# =============================================================================

import datetime as dt
from sklearn.preprocessing import FunctionTransformer

# Current year for age calculations
CURRENT_YEAR = dt.date.today().year


def year_to_age(x, *, current_year=CURRENT_YEAR):
    """Convert model-year → age in years."""
    return current_year - x


def age_odometer_product(X, *, current_year=CURRENT_YEAR):
    """
    X has two columns: [year, odometer].
    Returns one column: (age * odometer).
    Works whether X is a DataFrame or an ndarray.
    """
    if hasattr(X, "to_numpy"):                 # pandas -> ndarray
        X = X.to_numpy()

    yr, odo = X[:, 0], X[:, 1]
    age = current_year - yr
    return (age * odo).reshape(-1, 1)


# Pre-configured function transformers
age_transformer = FunctionTransformer(
    year_to_age,
    feature_names_out='one-to-one'
)

age_odometer_transformer = FunctionTransformer(
    age_odometer_product,
    feature_names_out=lambda _: ['age*odometer']
)


# =============================================================================
# MODEL TRAINING HELPERS
# =============================================================================

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def fit_gridsearch_pipeline(pipeline, param_grid, X_train, y_train, cv=3, verbose=False):
    """
    Fit a GridSearchCV pipeline and print results.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to optimize
    param_grid : dict
        Parameter grid for GridSearchCV
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=3
        Number of cross-validation folds
    verbose : bool, default=False
        Whether to print detailed information
    
    Returns:
    --------
    GridSearchCV : The fitted GridSearchCV object
    """
    
    if verbose:
        print("Starting GridSearchCV hyperparameter search...")
        
        # Print parameter grid details
        for param_name, param_values in param_grid.items():
            param_display_name = param_name.replace('__', ' ').replace('_', ' ').title()
            print(f"Testing {len(param_values)} {param_display_name} values: {param_values}")
        
        total_combinations = np.prod([len(values) for values in param_grid.values()])
        print(f"Total combinations: {total_combinations}")
        print(f"Using sample size: {len(X_train)} for hyperparameter tuning")
        print("\nFitting GridSearchCV...")
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Calculate RMSE for basic output
    y_pred = grid_search.predict(X_train)
    sample_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    
    if verbose:
        print("GridSearchCV completed!")
        
        # Print best parameters
        best_params = grid_search.best_params_
        print(f"\nBest parameters found:")
        for param_name, param_value in best_params.items():
            param_display_name = param_name.replace('__', ' ').replace('_', ' ').title()
            print(f"  {param_display_name}: {param_value}")
    else:
        # Simple output similar to pipeline1
        print("Training completed successfully!")
        
        # Extract and display all best parameters
        best_params = grid_search.best_params_
        for param_name, param_value in best_params.items():
            param_display_name = param_name.replace('__', ' ').replace('_', ' ').title()
            print(f"Best {param_display_name} selected by GridSearchCV: {param_value}")
    
    print(f"Sample RMSE: ${sample_rmse:,.2f}")
    
    return grid_search


def visualize_gridsearch_results(grid_search, figsize=(10, 6), title="GridSearchCV Performance", 
                                 score_type="RMSE", cmap='viridis_r'):
    """
    Create a heatmap visualization of GridSearchCV results across parameter grid.
    
    Parameters:
    -----------
    grid_search : GridSearchCV
        Fitted GridSearchCV object
    figsize : tuple, default=(10, 6)
        Figure size for the plot
    title : str, default="GridSearchCV Performance"
        Title for the heatmap
    score_type : str, default="RMSE"
        Type of score to display ("RMSE" converts from negative MSE)
    cmap : str, default='viridis_r'
        Colormap for the heatmap (reversed viridis so darker = better)
    
    Returns:
    --------
    dict : Dictionary containing performance statistics
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract GridSearchCV results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Get parameter names (remove 'param_' prefix)
    param_columns = [col for col in results_df.columns if col.startswith('param_')]
    
    if len(param_columns) != 2:
        print(f"Warning: This function is designed for 2-parameter grids. Found {len(param_columns)} parameters.")
        print(f"Parameters: {[col.replace('param_', '') for col in param_columns]}")
        return None
    
    # Create pivot table for heatmap
    param1, param2 = param_columns[0], param_columns[1]
    pivot_table = results_df.pivot_table(
        values='mean_test_score', 
        index=param1, 
        columns=param2
    )
    
    # Convert scores if needed
    if score_type.upper() == "RMSE":
        # Convert negative MSE to positive RMSE
        pivot_table_display = np.sqrt(-pivot_table)
        score_label = 'RMSE ($)'
        fmt = '.0f'
    else:
        pivot_table_display = pivot_table
        score_label = score_type
        fmt = '.4f'
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_table_display, 
        annot=True, 
        fmt=fmt,
        cmap=cmap,
        cbar_kws={'label': score_label}
    )
    
    # Clean up parameter names for labels
    param1_clean = param1.replace('param_', '').replace('__', ' ').replace('_', ' ').title()
    param2_clean = param2.replace('param_', '').replace('__', ' ').replace('_', ' ').title()
    
    plt.title(f'{title}: {score_type} Across Parameter Grid')
    plt.xlabel(param2_clean)
    plt.ylabel(param1_clean)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print statistics
    print(f"\nBest parameter combination:")
    best_params = grid_search.best_params_
    for param_name, param_value in best_params.items():
        param_display_name = param_name.replace('__', ' ').replace('_', ' ').title()
        print(f"  {param_display_name}: {param_value}")
    
    if score_type.upper() == "RMSE":
        best_score = np.sqrt(-grid_search.best_score_)
        print(f"  Best CV {score_type}: ${best_score:,.2f}")
        
        # Show performance range
        min_score = pivot_table_display.min().min()
        max_score = pivot_table_display.max().max()
        print(f"\nPerformance range:")
        print(f"  Best {score_type}: ${min_score:,.2f}")
        print(f"  Worst {score_type}: ${max_score:,.2f}")
        print(f"  Improvement: ${max_score - min_score:,.2f} ({((max_score - min_score) / max_score * 100):.1f}%)")
    else:
        print(f"  Best CV {score_type}: {grid_search.best_score_:.4f}")
        
        # Show performance range  
        min_score = pivot_table_display.min().min()
        max_score = pivot_table_display.max().max()
        print(f"\nPerformance range:")
        print(f"  Best {score_type}: {min_score:.4f}")
        print(f"  Worst {score_type}: {max_score:.4f}")
        print(f"  Range: {max_score - min_score:.4f}")
    
    # Return statistics
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'pivot_table': pivot_table_display,
        'param_names': [param1_clean, param2_clean]
    }


