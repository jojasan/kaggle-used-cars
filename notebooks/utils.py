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
    feature_names_out=lambda transformer, input_features: ['age*odometer']
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


# =============================================================================
# HOLDOUT EVALUATION HELPERS
# =============================================================================

from sklearn.metrics import mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred, model_name):
    """
    Calculate performance metrics for a model.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like  
        Predicted target values
    model_name : str
        Name of the model for identification
    
    Returns:
    --------
    dict : Dictionary containing model performance metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }


def display_holdout_evaluation_results(y_test, y_pred1, y_pred2, pipeline1, pipeline2, 
                                       model1_name="Pipeline 1 (Ridge)", 
                                       model2_name="Pipeline 2 (Lasso→Ridge)",
                                       X_test_sample=None):
    """
    Display comprehensive holdout evaluation results comparing two models.
    
    Parameters:
    -----------
    y_test : array-like
        True test set target values
    y_pred1 : array-like
        Predictions from first model
    y_pred2 : array-like
        Predictions from second model
    pipeline1 : sklearn.pipeline.Pipeline
        First trained pipeline
    pipeline2 : sklearn.pipeline.Pipeline
        Second trained pipeline
    model1_name : str, default="Pipeline 1 (Ridge)"
        Name for first model
    model2_name : str, default="Pipeline 2 (Lasso→Ridge)"
        Name for second model
    X_test_sample : array-like, optional
        Sample of test features for feature analysis (uses first row if not provided)
    
    Returns:
    --------
    pandas.DataFrame : DataFrame containing the metrics comparison
    """
    import pandas as pd
    
    print("Evaluating models on holdout test set...\n")
    
    # Calculate metrics for both models
    metrics1 = calculate_metrics(y_test, y_pred1, model1_name)
    metrics2 = calculate_metrics(y_test, y_pred2, model2_name)
    
    # Display results
    results_df = pd.DataFrame([metrics1, metrics2])
    print("📊 HOLDOUT TEST SET PERFORMANCE")
    print("=" * 50)
    print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<8}")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<25} ${row['RMSE']:<11,.0f} ${row['MAE']:<11,.0f} {row['R²']:<8.4f}")

    print("\n🏆 MODEL COMPARISON")
    print("=" * 30)
    rmse_diff = metrics1['RMSE'] - metrics2['RMSE']
    mae_diff = metrics1['MAE'] - metrics2['MAE']
    r2_diff = metrics2['R²'] - metrics1['R²']  # Higher R² is better

    if rmse_diff > 0:
        winner = model2_name.split()[0] + " " + model2_name.split()[1]  # Extract "Pipeline 2"
        rmse_improvement = f"${rmse_diff:,.0f} better RMSE"
    else:
        winner = model1_name.split()[0] + " " + model1_name.split()[1]  # Extract "Pipeline 1"
        rmse_improvement = f"${abs(rmse_diff):,.0f} better RMSE"

    print(f"Best RMSE: {winner} ({rmse_improvement})")
    print(f"MAE difference: ${mae_diff:+,.0f} ({model1_name.split()[0]} {model1_name.split()[1]} vs {model2_name.split()[0]} {model2_name.split()[1]})")
    print(f"R² difference: {r2_diff:+.4f} ({model2_name.split()[0]} {model2_name.split()[1]} vs {model1_name.split()[0]} {model1_name.split()[1]})")

    # Additional insights
    print(f"\n📈 PERFORMANCE INSIGHTS")
    print("=" * 30)
    avg_price = y_test.mean()
    print(f"Average test set price: ${avg_price:,.0f}")
    print(f"{model1_name.split()[0]} {model1_name.split()[1]} RMSE as % of avg price: {(metrics1['RMSE']/avg_price)*100:.2f}%")
    print(f"{model2_name.split()[0]} {model2_name.split()[1]} RMSE as % of avg price: {(metrics2['RMSE']/avg_price)*100:.2f}%")

    # Check if feature selection made a difference
    if X_test_sample is None:
        # Use a small sample for feature analysis if not provided
        X_test_sample = y_test.head(1) if hasattr(y_test, 'head') else [y_test[0]]
        
    try:
        n_features_original = pipeline1.named_steps['preprocessor'].transform(X_test_sample).shape[1]
        
        if 'selector' in pipeline2.named_steps:
            n_features_selected = pipeline2.named_steps['selector'].transform(
                pipeline2.named_steps['preprocessor'].transform(X_test_sample)
            ).shape[1]
            
            print(f"\n🔍 FEATURE SELECTION ANALYSIS")
            print("=" * 35)
            print(f"Original features: {n_features_original:,}")
            print(f"Selected features: {n_features_selected:,}")
            print(f"Features removed: {n_features_original - n_features_selected:,} ({((n_features_original - n_features_selected)/n_features_original)*100:.1f}%)")
        else:
            print(f"\n🔍 FEATURE ANALYSIS")
            print("=" * 20)
            print(f"Total features used: {n_features_original:,}")
            
    except Exception as e:
        print(f"\n⚠️  Feature analysis could not be completed: {str(e)}")
    
    return results_df


def get_feature_importance(pipeline, model_name, top_n=15):
    """
    Extract and rank feature importance from trained pipeline
    """
    print(f"\n🔍 Analyzing {model_name} Feature Importance...")
    
    # Get feature names from preprocessor
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    if 'selector' in pipeline.named_steps:
        # For Pipeline 2: get selected features and their coefficients
        selector = pipeline.named_steps['selector']
        selected_features_mask = selector.get_support()
        selected_feature_names = feature_names[selected_features_mask]
        coefficients = pipeline.named_steps['model'].coef_
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': selected_feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        })
        
        print(f"  ✓ Features after selection: {len(selected_feature_names):,} (from {len(feature_names):,})")
        
    else:
        # For Pipeline 1: all features and their coefficients
        coefficients = pipeline.named_steps['model'].coef_
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        })
        
        print(f"  ✓ Total features: {len(feature_names):,}")
    
    # Sort by absolute coefficient value and get top N
    importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
    top_features = importance_df.head(top_n).copy()
    
    print(f"  ✓ Top {top_n} most important features identified")
    
    return top_features, importance_df


def display_feature_importance_comparison(pipeline1_top, pipeline2_top, model1_name, model2_name, top_n=15):
    """
    Displays side-by-side feature importance and provides a comparison.
    """
    
    def display_top_features_table(features_df, model_name, top_n):
        """Helper to print a formatted table of top features."""
        print(f"\n🏆 {model_name} - Top {top_n} Features:")
        print("-" * 50)
        print(f"{'Rank':<4} {'Feature':<35} {'Coefficient':<12} {'Abs Value':<10}")
        print("-" * 50)
        for i, (_, row) in enumerate(features_df.iterrows(), 1):
            print(f"{i:<4} {row['Feature'][:34]:<35} {row['Coefficient']:<12.2f} {row['Abs_Coefficient']:<10.2f}")

    print(f"\n" + "="*60)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("="*60)
    
    display_top_features_table(pipeline1_top, model1_name, top_n)
    display_top_features_table(pipeline2_top, model2_name, top_n)

    # Compare overlap between top features
    pipeline1_top_features = set(pipeline1_top['Feature'].tolist())
    pipeline2_top_features = set(pipeline2_top['Feature'].tolist())
    common_features = pipeline1_top_features.intersection(pipeline2_top_features)
    unique_to_p1 = pipeline1_top_features - pipeline2_top_features
    unique_to_p2 = pipeline2_top_features - pipeline1_top_features

    print(f"\n📊 TOP FEATURES COMPARISON")
    print("="*40)
    print(f"Features in both top {top_n}: {len(common_features)} ({len(common_features)/top_n*100:.1f}%)")
    print(f"Unique to {model1_name}: {len(unique_to_p1)}")
    print(f"Unique to {model2_name}: {len(unique_to_p2)}")

    if common_features:
        print(f"\nCommon important features:")
        for feature in sorted(common_features):
            print(f"  • {feature}")


# Analyze feature types in top features
def analyze_feature_types(top_features_df, model_name):
    feature_types = {
        'Age-related': 0,
        'Odometer-related': 0, 
        'Manufacturer': 0,
        'Model': 0,
        'Condition': 0,
        'Type': 0,
        'Other': 0
    }
    
    for feature in top_features_df['Feature']:
        if 'age' in feature.lower():
            feature_types['Age-related'] += 1
        elif 'odometer' in feature.lower():
            feature_types['Odometer-related'] += 1
        elif 'manufacturer' in feature.lower():
            feature_types['Manufacturer'] += 1
        elif 'model' in feature.lower():
            feature_types['Model'] += 1
        elif 'condition' in feature.lower():
            feature_types['Condition'] += 1
        elif 'type' in feature.lower():
            feature_types['Type'] += 1
        else:
            feature_types['Other'] += 1
    
    print(f"\n{model_name} feature categories:")
    for category, count in feature_types.items():
        if count > 0:
            print(f"  {category}: {count}")


def display_comprehensive_feature_analysis(pipeline1_top, pipeline2_top, pipeline1_all, pipeline2_all,
                                            model1_name="Pipeline 1 (Ridge)", 
                                            model2_name="Pipeline 2 (Lasso→Ridge)", 
                                            top_n=15):
    """
    Display comprehensive feature importance analysis for two pipelines including:
    - Top N features for each model in formatted tables
    - Overlap comparison between models
    - Feature insights and categorization
    - Key insights and summary
    
    Parameters:
    -----------
    pipeline1_top : pandas.DataFrame
        Top N features from pipeline 1 with columns: Feature, Coefficient, Abs_Coefficient
    pipeline2_top : pandas.DataFrame
        Top N features from pipeline 2 with columns: Feature, Coefficient, Abs_Coefficient  
    pipeline1_all : pandas.DataFrame
        All features from pipeline 1
    pipeline2_all : pandas.DataFrame
        All features from pipeline 2
    model1_name : str, default="Pipeline 1 (Ridge)"
        Name for first model
    model2_name : str, default="Pipeline 2 (Lasso→Ridge)"
        Name for second model
    top_n : int, default=15
        Number of top features to display and analyze
    """
    
    print(f"\n" + "="*60)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("="*60)

    # Display Pipeline 1 top features
    print(f"\n🏆 {model1_name} - Top {top_n} Features:")
    print("-" * 50)
    print(f"{'Rank':<4} {'Feature':<35} {'Coefficient':<12} {'Abs Value':<10}")
    print("-" * 50)
    for i, (_, row) in enumerate(pipeline1_top.iterrows(), 1):
        print(f"{i:<4} {row['Feature'][:34]:<35} {row['Coefficient']:<12.2f} {row['Abs_Coefficient']:<10.2f}")

    # Display Pipeline 2 top features
    print(f"\n🏆 {model2_name} - Top {top_n} Features:")
    print("-" * 50)
    print(f"{'Rank':<4} {'Feature':<35} {'Coefficient':<12} {'Abs Value':<10}")
    print("-" * 50)
    for i, (_, row) in enumerate(pipeline2_top.iterrows(), 1):
        print(f"{i:<4} {row['Feature'][:34]:<35} {row['Coefficient']:<12.2f} {row['Abs_Coefficient']:<10.2f}")

    # Compare overlap between top features
    pipeline1_top_features = set(pipeline1_top['Feature'].tolist())
    pipeline2_top_features = set(pipeline2_top['Feature'].tolist())
    common_features = pipeline1_top_features.intersection(pipeline2_top_features)
    unique_to_p1 = pipeline1_top_features - pipeline2_top_features
    unique_to_p2 = pipeline2_top_features - pipeline1_top_features

    print(f"\n📊 TOP FEATURES COMPARISON")
    print("="*40)
    print(f"Features in both top {top_n}: {len(common_features)} ({len(common_features)/top_n*100:.1f}%)")
    print(f"Unique to {model1_name.split()[0]} {model1_name.split()[1]}: {len(unique_to_p1)}")
    print(f"Unique to {model2_name.split()[0]} {model2_name.split()[1]}: {len(unique_to_p2)}")

    if common_features:
        print(f"\nCommon important features:")
        for feature in sorted(common_features):
            print(f"  • {feature}")

    print(f"\n💡 FEATURE INSIGHTS")
    print("="*30)

    analyze_feature_types(pipeline1_top, model1_name.split()[0] + " " + model1_name.split()[1])
    analyze_feature_types(pipeline2_top, model2_name.split()[0] + " " + model2_name.split()[1])
    
    # Key insights summary
    print(f"\n🎯 KEY INSIGHTS")
    print("="*20)
    print(f"• Both models agree on {len(common_features)} out of {top_n} most important features")
    print(f"• Feature selection (Lasso) reduced features by {((len(pipeline1_all) - len(pipeline2_all))/len(pipeline1_all))*100:.1f}%")
    print(f"• The most important feature in {model1_name.split()[0]} {model1_name.split()[1]}: '{pipeline1_top.iloc[0]['Feature'][:50]}{'...' if len(pipeline1_top.iloc[0]['Feature']) > 50 else ''}'")
    print(f"• The most important feature in {model2_name.split()[0]} {model2_name.split()[1]}: '{pipeline2_top.iloc[0]['Feature'][:50]}{'...' if len(pipeline2_top.iloc[0]['Feature']) > 50 else ''}'")

    # Show coefficient magnitude comparison
    p1_max_coeff = pipeline1_top['Abs_Coefficient'].max()
    p2_max_coeff = pipeline2_top['Abs_Coefficient'].max()
    print(f"• Largest coefficient magnitude - {model1_name.split()[0]} {model1_name.split()[1]}: {p1_max_coeff:.2f}, {model2_name.split()[0]} {model2_name.split()[1]}: {p2_max_coeff:.2f}")

    # Summary of what drives predictions
    print(f"\n🚗 WHAT DRIVES CAR PRICE PREDICTIONS:")
    print("="*40)

    # Extract and summarize most important feature types
    important_categories = []
    for feature in pipeline1_top.head(5)['Feature']:
        if 'manufacturer' in feature.lower():
            important_categories.append("Vehicle Manufacturer")
        elif 'model' in feature.lower():
            important_categories.append("Vehicle Model")
        elif 'age' in feature.lower():
            important_categories.append("Vehicle Age")
        elif 'odometer' in feature.lower():
            important_categories.append("Mileage/Usage")
        elif 'condition' in feature.lower():
            important_categories.append("Vehicle Condition")
        elif 'type' in feature.lower():
            important_categories.append("Vehicle Type")

    # Remove duplicates while preserving order
    unique_categories = list(dict.fromkeys(important_categories))
    for i, category in enumerate(unique_categories[:3], 1):
        print(f"{i}. {category}")

    print("\nBoth models consistently identify these factors as the primary drivers of vehicle pricing.")