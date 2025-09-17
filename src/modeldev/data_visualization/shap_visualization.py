import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from src.modeldev.data_preprocessing.data_preprocessor import DataPreprocessor
import json
import numpy as np
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import json
from src.modeldev.data_preprocessing.data_preprocessor import DataPreprocessor
import matplotlib.pyplot as plt
def get_data():
    """
    Helper function to load data from content_fetcher.
    """
    try:
        data_preprocessor = DataPreprocessor(path='../../../data/input/merged_output_1m_2_202506_new_yearly.csv')
        df = data_preprocessor.preprocess_data()
        print(df.Y[df.Y==1].sum())
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(df):
    """
    Prepares the dataframe by dropping unnecessary columns and creating a date column.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        tuple: A tuple containing the prepared dataframe, features list, and target column name.
    """
    # Assuming the target variable is 'Y'
    target_column = 'Y'

    # Convert year and month into a single date column for chronological sorting
    df['date'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-' + df['month'].astype(int).astype(str),
                                format='%Y-%m')
    df = df.sort_values(by='date').reset_index(drop=True)

    # Define features (X) and target (y)
    features = [col for col in df.columns if col not in ['U3_company_number', 'year', 'month', 'date', target_column]]

    return df, features, target_column


def split_data(df, features, target_column, test_year=2020):
    """
    Splits the data into training and test sets based on years.
    Modified for yearly prediction.

    Args:
        df (pd.DataFrame): The prepared dataframe.
        features (list): The list of feature column names.
        target_column (str): The name of the target column.
        test_year (int): The year to use as test set.

    Returns:
        dict: A dictionary containing the split dataframes and series.
    """
    # Test set: the specified test year
    test_data = df[df['year'] == test_year].copy()

    # Final training set: all years before the test year
    train_val_data = df[df['year'] < test_year].copy()

    splits = {
        'X_train_final': train_val_data[features],
        'y_train_final': train_val_data[target_column],
        'X_test': test_data[features],
        'y_test': test_data[target_column]
    }

    print(f"Final Model Training Data Shape: {splits['X_train_final'].shape}")
    print(f"Test Data Shape: {splits['X_test'].shape}")
    print(f"Training years: {sorted(train_val_data['year'].unique())}")
    print(f"Test year: {test_year}")

    return splits

def analyze_feature_importance_over_time(df, features, target_column, best_params,
                                         start_year=2015, end_year=2020,
                                         top_n_features=15, save_plots=True):
    """
    Analyzes feature importance using SHAP values with expanding windows over years.

    Args:
        df (pd.DataFrame): The prepared dataframe.
        features (list): The list of feature column names.
        target_column (str): The name of the target column.
        best_params (dict): The best hyperparameters to use for the model.
        start_year (int): The starting year for the analysis.
        end_year (int): The ending year for the analysis.
        top_n_features (int): Number of top features to track over time.
        save_plots (bool): Whether to save plots to files.

    Returns:
        dict: Dictionary containing SHAP values and feature importance rankings for each year.
    """
    print(f"\nStarting SHAP feature importance analysis from {start_year} to {end_year}...")

    # Initialize containers for results
    yearly_shap_values = {}
    yearly_feature_importance = {}
    yearly_models = {}

    test_years = list(range(start_year, end_year + 1))

    for test_year in test_years:
        print(f"\nAnalyzing year {test_year}...")

        # Define expanding training window (all data up to test year)
        train_data = df[df['year'] < test_year]
        test_data = df[df['year'] == test_year]

        if train_data.empty or test_data.empty:
            print(f"Skipping {test_year}: insufficient data")
            continue

        X_train = train_data[features]
        y_train = train_data[target_column]
        X_test = test_data[features]
        y_test = test_data[target_column]

        # Train model
        lgb_model = lgb.LGBMClassifier(**best_params)
        lgb_model.fit(X_train, y_train)
        yearly_models[test_year] = lgb_model

        # Calculate SHAP values for test set
        # Use TreeExplainer for LightGBM (faster and more accurate for tree models)
        explainer = shap.TreeExplainer(lgb_model)

        # For multiclass, we'll focus on class 1 predictions
        shap_values = explainer.shap_values(X_test)

        # For multiclass LightGBM, shap_values is a list of arrays (one per class)
        # We want class 1 SHAP values
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]  # Class 1 SHAP values
        else:
            shap_values_class1 = shap_values

        yearly_shap_values[test_year] = {
            'shap_values': shap_values_class1,
            'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                    list) else explainer.expected_value,
            'feature_names': features,
            'X_test': X_test
        }

        # Calculate mean absolute SHAP values for feature importance ranking
        mean_abs_shap = np.mean(np.abs(shap_values_class1), axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        yearly_feature_importance[test_year] = feature_importance_df

        print(f"Top 5 features for {test_year}:")
        for i, row in feature_importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    # Create comprehensive visualizations
    if save_plots:
        create_feature_importance_plots(yearly_feature_importance, yearly_shap_values,
                                        top_n_features, start_year, end_year)

    return {
        'yearly_shap_values': yearly_shap_values,
        'yearly_feature_importance': yearly_feature_importance,
        'yearly_models': yearly_models
    }
def run_expanding_backtest(df, features, target_column, best_params, start_year=2023, start_month=1, end_year=2025, end_month=6):
    """
    Performs expanding window backtesting and returns the average AUC.
    Modified for yearly prediction.

    Args:
        df (pd.DataFrame): The prepared dataframe.
        features (list): The list of feature column names.
        target_column (str): The name of the target column.
        best_params (dict): The best hyperparameters to use for the model.
        start_year (int): The starting year for the backtest.
        end_year (int): The ending year for the backtest.

    Returns:
        tuple: The mean AUC score, mean PR-AUC score, and results dataframe.
    """
    print("\nStarting expanding window backtesting...")

    # Generate a list of test years
    date_range = pd.date_range(start=f'{start_year}-{start_month}', end=f'{end_year}-{end_month}', freq='MS')

    auc_results = []
    prauc_results = []
    results = []

    for test_date in date_range:
        if str(test_date) == '2023-12-01 00:00:00':
            print('here')
        # Define the training window (all data up to the test year)
        train_data = df[df['date'] < test_date]
        filter_range = test_date + pd.DateOffset(years=1)
        # Define the test window (the current test year)
        test_data = df[(df["date"] >= test_date) & (df["date"] < filter_range)]

        if not test_data.empty and not train_data.empty:
            X_train = train_data[features]
            y_train = train_data[target_column]
            X_test = test_data[features]
            y_test = test_data[target_column]

            # Train model with best hyperparameters
            lgb_model = lgb.LGBMClassifier(**best_params)
            lgb_model.fit(X_train, y_train)

            # Predict probabilities
            y_pred_proba = lgb_model.predict_proba(X_test)

            # Calculate AUC and store it
            if len(np.unique(y_test)) > 1:  # Check if we have multiple classes
                mask = y_test != 2
                if mask.any():  # Check if we have class 0 or 1 samples
                    auc_score = roc_auc_score(y_test[mask], y_pred_proba[mask][:, 1])

                    precision, recall, thresholds = precision_recall_curve(y_test[mask], y_pred_proba[mask][:, 1])
                    pr_auc = auc(recall, precision)

                    auc_results.append(auc_score)
                    prauc_results.append(pr_auc)

                    results.append({
                        'year': test_date,
                        'auc_score': auc_score,
                        'pr_auc': pr_auc,
                        'train_size': len(train_data),
                        'test_size': len(test_data),
                        'test_positives': y_test[mask].sum(),
                        'test_samples': len(y_test[mask])
                    })
                    print(f"Backtest for {test_date}: AUC = {auc_score:.4f}, PR-AUC = {pr_auc:.4f}")
                else:
                    print(f"Skipping backtest for {test_date}: No class 0 or 1 samples in test data.")
            else:
                print(f"Skipping backtest for {test_date}: Not enough classes in test data.")
        else:
            print(f"Skipping backtest for {test_date}: No test data or training data available.")

    if results:
        results_df = pd.DataFrame(results)
        mean_auc = np.mean(auc_results)
        mean_prauc = np.mean(prauc_results)
        print(f"\nExpanding Window Backtest Mean AUC: {mean_auc:.4f}")
        print(f"Expanding Window Backtest Mean PR-AUC: {mean_prauc:.4f}")
        return mean_auc, mean_prauc, results_df
    else:
        print("\nNo backtest results were calculated.")
        return None, None, pd.DataFrame()

def create_feature_importance_plots(yearly_feature_importance, yearly_shap_values,
                                    top_n_features=15, start_year=2015, end_year=2020):
    """
    Creates comprehensive SHAP feature importance visualizations.

    Args:
        yearly_feature_importance (dict): Feature importance data by year.
        yearly_shap_values (dict): SHAP values data by year.
        top_n_features (int): Number of top features to display.
        start_year (int): Starting year for analysis.
        end_year (int): Ending year for analysis.
    """

    # 1. Feature Importance Evolution Heatmap
    plt.figure(figsize=(16, 10))

    # Get all unique features that appear in top N across all years
    all_top_features = set()
    for year_data in yearly_feature_importance.values():
        all_top_features.update(year_data.head(top_n_features)['feature'].tolist())

    all_top_features = list(all_top_features)
    years = sorted(yearly_feature_importance.keys())

    # Create importance matrix
    importance_matrix = []
    for year in years:
        year_importance = yearly_feature_importance[year].set_index('feature')['importance']
        row = [year_importance.get(feature, 0) for feature in all_top_features]
        importance_matrix.append(row)

    importance_df = pd.DataFrame(importance_matrix, index=years, columns=all_top_features)

    # Plot heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(importance_df.T, annot=False, cmap='viridis', cbar_kws={'label': 'Mean |SHAP Value|'})
    plt.title('Feature Importance Evolution Over Years')
    plt.xlabel('Year')
    plt.ylabel('Features')

    # 2. Top Features Ranking Over Time
    plt.subplot(2, 2, 2)

    # Track rank of top 10 features over time
    consistent_top_features = list(all_top_features)[:10]

    for feature in consistent_top_features:
        ranks = []
        for year in years:
            year_df = yearly_feature_importance[year]
            try:
                rank = year_df[year_df['feature'] == feature].index[0] + 1
            except:
                rank = len(year_df) + 1  # If feature not found, put at bottom
            ranks.append(rank)

        plt.plot(years, ranks, marker='o', label=feature, alpha=0.7)

    plt.title('Feature Ranking Evolution')
    plt.xlabel('Year')
    plt.ylabel('Rank (1 = Most Important)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_yaxis()  # Lower rank numbers at top
    plt.grid(True, alpha=0.3)

    # 3. Feature Importance Trends
    plt.subplot(2, 2, 3)

    # Show importance values for top 5 most consistent features
    top_consistent_features = list(all_top_features)[:5]

    for feature in top_consistent_features:
        importance_values = []
        for year in years:
            year_df = yearly_feature_importance[year]
            importance = year_df[year_df['feature'] == feature]['importance'].iloc[0] if feature in year_df[
                'feature'].values else 0
            importance_values.append(importance)

        plt.plot(years, importance_values, marker='s', label=feature, linewidth=2)

    plt.title('Top 5 Features Importance Trends')
    plt.xlabel('Year')
    plt.ylabel('Mean |SHAP Value|')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. SHAP Summary for Latest Year
    plt.subplot(2, 2, 4)

    latest_year = max(years)
    latest_shap_data = yearly_shap_values[latest_year]

    # Create a simple bar plot of feature importance for latest year
    latest_importance = yearly_feature_importance[latest_year].head(10)

    plt.barh(range(len(latest_importance)), latest_importance['importance'])
    plt.yticks(range(len(latest_importance)), latest_importance['feature'])
    plt.title(f'Top 10 Features - Year {latest_year}')
    plt.xlabel('Mean |SHAP Value|')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'shap_feature_importance_analysis_{start_year}_{end_year}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def create_detailed_shap_plots(yearly_shap_values, year_to_plot=None):
    """
    Creates detailed SHAP plots for a specific year.

    Args:
        yearly_shap_values (dict): SHAP values data by year.
        year_to_plot (int): Specific year to create detailed plots for. If None, uses latest year.
    """
    if not yearly_shap_values:
        print("No SHAP values available for plotting.")
        return

    if year_to_plot is None:
        year_to_plot = max(yearly_shap_values.keys())

    if year_to_plot not in yearly_shap_values:
        print(f"Year {year_to_plot} not found in SHAP values.")
        return

    shap_data = yearly_shap_values[year_to_plot]
    shap_values = shap_data['shap_values']
    X_test = shap_data['X_test']

    print(f"Creating detailed SHAP plots for year {year_to_plot}...")

    # 1. SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
    plt.title(f'SHAP Feature Importance - Year {year_to_plot}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_bar_{year_to_plot}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title(f'SHAP Summary Plot - Year {year_to_plot}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_beeswarm_{year_to_plot}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. SHAP Waterfall Plot for first prediction
    if len(shap_values) > 0:
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=shap_data['base_value'],
                data=X_test.iloc[0].values,
                feature_names=X_test.columns.tolist()
            ),
            show=False,
            max_display=15
        )
        plt.title(f'SHAP Waterfall Plot - First Prediction, Year {year_to_plot}')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_{year_to_plot}.png', dpi=300, bbox_inches='tight')
        plt.show()


def track_feature_stability(yearly_feature_importance, top_n=10):
    """
    Analyzes which features are consistently important across years.

    Args:
        yearly_feature_importance (dict): Feature importance data by year.
        top_n (int): Number of top features to consider for each year.

    Returns:
        pd.DataFrame: Feature stability analysis results.
    """
    print(f"\nAnalyzing feature stability across years...")

    # Track how often each feature appears in top N
    feature_appearances = defaultdict(int)
    feature_avg_rank = defaultdict(list)
    feature_avg_importance = defaultdict(list)

    years = sorted(yearly_feature_importance.keys())

    for year in years:
        top_features = yearly_feature_importance[year].head(top_n)

        for idx, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            rank = idx + 1

            feature_appearances[feature] += 1
            feature_avg_rank[feature].append(rank)
            feature_avg_importance[feature].append(importance)

    # Create stability analysis dataframe
    stability_results = []
    for feature in feature_appearances.keys():
        stability_results.append({
            'feature': feature,
            'appearances_in_top_n': feature_appearances[feature],
            'appearance_rate': feature_appearances[feature] / len(years),
            'avg_rank': np.mean(feature_avg_rank[feature]),
            'avg_importance': np.mean(feature_avg_importance[feature]),
            'importance_std': np.std(feature_avg_importance[feature]) if len(feature_avg_importance[feature]) > 1 else 0
        })

    stability_df = pd.DataFrame(stability_results)
    stability_df = stability_df.sort_values(['appearance_rate', 'avg_importance'], ascending=[False, False])

    print(f"\nMost Stable Features (appearing in top {top_n} across years):")
    print(stability_df.head(10).to_string(index=False))

    # Save results
    stability_df.to_csv(f'feature_stability_analysis_{start_year}_{end_year}.csv', index=False)

    return stability_df


def plot_feature_importance_trends(yearly_feature_importance, features_to_plot=None,
                                   start_year=2015, end_year=2020, save_path=None):
    """
    Plots feature importance trends over time for specific features.

    Args:
        yearly_feature_importance (dict): Feature importance data by year.
        features_to_plot (list): List of features to plot. If None, uses top 10 from latest year.
        start_year (int): Starting year for the plot.
        end_year (int): Ending year for the plot.
        save_path (str): Path to save the plot.
    """
    years = sorted(yearly_feature_importance.keys())

    if features_to_plot is None:
        # Use top 10 features from the latest year
        latest_year = max(years)
        features_to_plot = yearly_feature_importance[latest_year].head(10)['feature'].tolist()

    plt.figure(figsize=(15, 10))

    # Create subplot for each feature
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    for i, feature in enumerate(features_to_plot):
        plt.subplot(n_rows, n_cols, i + 1)

        importance_values = []
        plot_years = []

        for year in years:
            year_df = yearly_feature_importance[year]
            if feature in year_df['feature'].values:
                importance = year_df[year_df['feature'] == feature]['importance'].iloc[0]
                importance_values.append(importance)
                plot_years.append(year)

        if importance_values:
            plt.plot(plot_years, importance_values, marker='o', linewidth=2, markersize=4)
            plt.title(f'{feature}', fontsize=10)
            plt.xlabel('Year')
            plt.ylabel('Importance')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

    plt.suptitle('Feature Importance Trends Over Years', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature trends plot saved to {save_path}")

    plt.show()


def create_shap_heatmap(yearly_shap_values, top_features=15, save_path=None):
    """
    Creates a heatmap showing average SHAP values for top features across years.

    Args:
        yearly_shap_values (dict): SHAP values data by year.
        top_features (int): Number of top features to include.
        save_path (str): Path to save the plot.
    """
    years = sorted(yearly_shap_values.keys())

    # Get all features and their average importance across all years
    all_feature_importance = defaultdict(list)

    for year in years:
        shap_values = yearly_shap_values[year]['shap_values']
        feature_names = yearly_shap_values[year]['feature_names']

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        for feature, importance in zip(feature_names, mean_abs_shap):
            all_feature_importance[feature].append(importance)

    # Calculate overall average importance
    overall_importance = {
        feature: np.mean(importances)
        for feature, importances in all_feature_importance.items()
    }

    # Get top features based on overall importance
    top_feature_names = sorted(overall_importance.keys(),
                               key=lambda x: overall_importance[x],
                               reverse=True)[:top_features]

    # Create heatmap data
    heatmap_data = []
    for year in years:
        if year in yearly_shap_values:
            shap_values = yearly_shap_values[year]['shap_values']
            feature_names = yearly_shap_values[year]['feature_names']

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_dict = dict(zip(feature_names, mean_abs_shap))

            row = [feature_dict.get(feature, 0) for feature in top_feature_names]
            heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=years, columns=top_feature_names)

    # Create the heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_df.T, annot=False, cmap='viridis',
                cbar_kws={'label': 'Mean |SHAP Value|'})
    plt.title('Feature Importance Heatmap Across Years')
    plt.xlabel('Year')
    plt.ylabel('Features')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP heatmap saved to {save_path}")

    plt.show()


# Modified main function to include SHAP analysis
def main_with_shap():
    """
    Main function that includes SHAP feature importance analysis.
    """
    print("Loading data...")
    df = get_data()

    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Configuration for yearly prediction
    windows_years = 10
    test_year = 2025 - windows_years
    expanding_time = f'{windows_years}yrs'

    # Step 1: Prepare the data
    df, features, target_column = prepare_data(df)

    # Step 2: Split the data
    splits = split_data(df, features, target_column, test_year=test_year)

    # Step 3: Load or tune hyperparameters
    try:
        with open(f'best_lgbm_params_yearly_{expanding_time}.json', 'r') as f:
            best_params = json.load(f)
        print("Loaded best parameters from yearly parameters file.")
    except FileNotFoundError:
        print("Best parameters file not found. Please run hyperparameter tuning first.")
        return

    best_params['num_threads'] = 4

    # Step 4: Run expanding window backtesting
    backtest_start_year = max(df['year'].min() + 2, test_year - 5)
    auc_mean, prauc_mean, results_df = run_expanding_backtest(
        df, features, target_column, best_params,
        start_year=backtest_start_year, end_year=test_year
    )

    # Step 5: SHAP Feature Importance Analysis
    print("\n" + "=" * 80)
    print("STARTING SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    shap_results = analyze_feature_importance_over_time(
        df, features, target_column, best_params,
        start_year=backtest_start_year,
        end_year=test_year,
        top_n_features=20,
        save_plots=True
    )

    # Step 6: Create additional SHAP visualizations
    print("\nCreating additional SHAP visualizations...")

    # Feature stability analysis
    stability_df = track_feature_stability(
        shap_results['yearly_feature_importance'],
        top_n=15
    )

    # Feature importance trends
    plot_feature_importance_trends(
        shap_results['yearly_feature_importance'],
        features_to_plot=stability_df.head(8)['feature'].tolist(),
        start_year=backtest_start_year,
        end_year=test_year,
        save_path=f'feature_trends_{backtest_start_year}_{test_year}.png'
    )

    # SHAP heatmap
    create_shap_heatmap(
        shap_results['yearly_shap_values'],
        top_features=15,
        save_path=f'shap_heatmap_{backtest_start_year}_{test_year}.png'
    )

    # Detailed SHAP plots for latest year
    create_detailed_shap_plots(shap_results['yearly_shap_values'])

    # Step 7: Train final model and make predictions
    train_and_predict(splits['X_train_final'], splits['y_train_final'],
                      splits['X_test'], splits['y_test'],
                      best_params)

    return shap_results


# Example usage:
if __name__ == "__main__":
    # Run the main function with SHAP analysis
    shap_results = main_with_shap()

    # You can also run individual SHAP analyses:
    # analyze_feature_importance_over_time(df, features, target_column, best_params)
    # create_detailed_shap_plots(shap_results['yearly_shap_values'], year_to_plot=2020)