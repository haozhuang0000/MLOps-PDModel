from src.mlops.database.db_connection import MySQLConnection
import os

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
from datetime import datetime

def get_data(path):
    """
    Helper function to load data from content_fetcher.
    """
    try:
        data_preprocessor = DataPreprocessor(path=path)
        df = data_preprocessor.preprocess_data()
        df['Y'] = df.groupby('U3_company_number')['Y'].shift(-1)
        df = df.dropna(subset=['Y'])
        df['Y'] = df['Y'].astype(int)
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


def split_data(df, features, target_column, test_year=2025, test_month=6):
    """
    Splits the data into training and test sets based on dates.

    Args:
        df (pd.DataFrame): The prepared dataframe.
        features (list): The list of feature column names.
        target_column (str): The name of the target column.

    Returns:
        dict: A dictionary containing the split dataframes and series.
    """
    # Test set: the last data point for each company, 2025.06
    test_data = df[(df['year'] == test_year) & (df['month'] == test_month)].copy()
    date_ = f'{test_year}-{test_month}-01'
    # Final training set: up to 2025.05
    train_val_data = df[df['date'] < date_].copy()

    splits = {
        'X_train_final': train_val_data[features],
        'y_train_final': train_val_data[target_column],
        'X_test': test_data[features],
        'y_test': test_data[target_column]
    }

    print(f"Final Model Training Data Shape: {splits['X_train_final'].shape}")
    print(f"Test Data Shape: {splits['X_test'].shape}")

    return splits


def tune_hyperparameters(X_train_full, y_train_full, expanding_time):
    """
    Tunes LightGBM hyperparameters using Optuna and TimeSeriesSplit for an expanding window.
    Optimizes for class 1 prediction in multi-class scenario.
    Args:
        X_train_full (pd.DataFrame): The full training set for tuning.
        y_train_full (pd.Series): The full target series for tuning.
    Returns:
        dict: A dictionary containing best hyperparameters, best AUC, and best PR-AUC.
    """
    print("\nStarting hyperparameter tuning with Optuna and expanding windows...")
    num_classes = y_train_full.nunique()
    print(f"Data has {num_classes} classes: {sorted(y_train_full.unique())}")
    print("Optimizing for class 0 vs 1 (excluding class 2)")

    # Always use multiclass for LightGBM when we have 3 classes
    objective = 'multiclass'
    metric = 'multi_logloss'
    num_class = num_classes

    def optuna_objective(trial):  # Fixed the function name
        """
        Objective function for Optuna to optimize hyperparameters using cross-validation.
        Focuses on class 1 prediction performance.
        """
        param = {
            'objective': objective,
            'metric': metric,
            'num_class': num_class,
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'random_state': 42,
            'num_threads': 4,
            'verbose': -1  # Suppress LightGBM output
        }

        auc_scores = []
        pr_auc_scores = []

        # Use TimeSeriesSplit to create expanding windows
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full)):
            X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
            y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

            if len(np.unique(y_train)) < 3:
                print(f"Skipping fold {fold + 1}: only one class in training data")
                continue
            # Check if we have samples after filtering out class 2
            mask = y_val != 2
            if not mask.any():
                print(f"Warning: No class 0 or 1 samples in validation fold {fold + 1}, skipping...")
                continue

            lgb_model = lgb.LGBMClassifier(**param)
            lgb_model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

            # Get probabilities for all classes
            y_pred_proba = lgb_model.predict_proba(X_val)

            # Filter out class 2 samples (only use class 0 and 1)
            mask = y_val != 2
            y_val_filtered = y_val[mask]
            y_pred_proba_filtered = y_pred_proba[mask]

            # Calculate AUC for class 0 vs class 1 (excluding class 2)
            if len(y_val_filtered) > 0 and len(np.unique(y_val_filtered)) > 1:  # Check if both classes present
                auc_score = roc_auc_score(y_val_filtered, y_pred_proba_filtered[:, 1])

                precision, recall, thresholds = precision_recall_curve(y_val_filtered, y_pred_proba_filtered[:, 1])
                pr_auc = auc(recall, precision)

                # pr_auc = average_precision_score(y_val_filtered, y_pred_proba_filtered[:, 1])
                auc_scores.append(auc_score)
                pr_auc_scores.append(pr_auc)
            else:
                print(f"Warning: Only one class present in fold {fold + 1}, skipping AUC calculation...")

        # Return max AUC instead of mean
        if len(auc_scores) > 0:
            max_auc = np.max(auc_scores)
            max_pr_auc = np.max(pr_auc_scores)
            # Store PR-AUC in trial for later retrieval
            trial.set_user_attr("pr_auc", max_pr_auc)
            trial.set_user_attr("auc_scores", auc_scores)  # Store all scores for analysis
            trial.set_user_attr("pr_auc_scores", pr_auc_scores)
            return max_auc
        else:
            trial.set_user_attr("pr_auc", 0.0)
            trial.set_user_attr("auc_scores", [])
            trial.set_user_attr("pr_auc_scores", [])
            return 0.0

    # Run the Optuna study
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=25)

    best_params = study.best_params.copy()
    best_params.update({
        'objective': objective,
        'metric': metric,
        'num_class': num_class,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1
    })

    # Get the best AUC and PR-AUC (now using max instead of mean)
    best_auc = study.best_value
    best_pr_auc = study.best_trial.user_attrs.get("pr_auc", 0.0)

    print("Hyperparameter tuning completed.")
    print(f"Best Max AUC (Class 0 vs 1, excluding Class 2): {best_auc:.4f}")
    print(f"Best Max PR-AUC (Class 0 vs 1, excluding Class 2): {best_pr_auc:.4f}")
    print(f"Best hyperparameters: {best_params}")

    results_dict = {
        'best_params': best_params,
        'best_auc': best_auc,
        'best_pr_auc': best_pr_auc
    }

    # Save to JSON file
    with open(f'hyperparameter_tuning_results_{expanding_time}.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    return best_params


# def train_and_predict(X_train, y_train, X_test, y_test, best_params):
#     """
#     Trains the final model on the full training data and makes predictions on the test set.
#
#     Args:
#         X_train (pd.DataFrame): Final training features.
#         y_train (pd.Series): Final training target.
#         X_test (pd.DataFrame): Test features.
#         y_test (pd.Series): Test target.
#         best_params (dict): Best hyperparameters from the tuning phase.
#     """
#     print("\nTraining final model...")
#     final_lgb_model = lgb.LGBMClassifier(**best_params)
#     final_lgb_model.fit(X_train, y_train)
#
#     print("Final model training completed.")
#
#     # Make predictions on the test set (2025.06)
#     print("\nMaking predictions on the 2025.06 test data...")
#     if not X_test.empty:
#         # Predict probabilities for all classes
#         y_pred_proba_test = final_lgb_model.predict_proba(X_test)
#
#         # Display the first few predictions
#         print("Predictions for the test set (first 5):")
#         for i in range(min(5, len(y_pred_proba_test))):
#             print(f"  Prediction {i + 1}: {y_pred_proba_test[i]}")
#
#         # Optional: Evaluate the performance if y_test has ground truth
#         if not y_test.empty and y_test.nunique() > 1:
#             if best_params['num_class'] is None or best_params['num_class'] <= 2:
#                 # Binary AUC
#                 test_auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
#             else:
#                 # Multi-class AUC using 'ovr'
#                 mask = y_test != 2
#                 test_auc = roc_auc_score(y_test[mask], y_pred_proba_test[mask][:, 1])
#             print(f"\nFinal Model Test Set AUC Score: {test_auc:.4f}")
#     else:
#         print("No test data found for 2025.06.")


def run_expanding_backtest(df, features, target_column, best_params, start_year=2024, start_month=6, end_year=2025,
                           end_month=6):
    """
    Performs expanding window backtesting and returns the average AUC.

    Args:
        df (pd.DataFrame): The prepared dataframe.
        features (list): The list of feature column names.
        target_column (str): The name of the target column.
        best_params (dict): The best hyperparameters to use for the model.
        start_year (int): The starting year for the backtest.
        start_month (int): The starting month for the backtest.
        end_year (int): The ending year for the backtest.
        end_month (int): The ending month for the backtest.

    Returns:
        float: The mean AUC score over all backtest windows.
    """
    print("\nStarting expanding window backtesting...")

    # Generate a list of test periods (e.g., '2024-06', '2024-07', ..., '2025-06')
    date_range = pd.date_range(start=f'{start_year}-{start_month}', end=f'{end_year}-{end_month}', freq='MS')

    auc_results = []
    prauc_results = []
    results = []
    for test_date in date_range:
        # Define the training window (all data up to the test month)
        train_data = df[df['date'] < test_date]

        # Define the test window (the current test month)
        test_data = df[df['date'] == test_date]

        if not test_data.empty:
            X_train = train_data[features]
            y_train = train_data[target_column]
            X_test = test_data[features]
            y_test = test_data[target_column]
            if len(np.unique(y_train)) < 2:
                print(f"Skipping test date {test_date}: only one class in training data")
                results.append({
                    # 'date': test_date.date,
                    'date': test_date.strftime('%Y-%m-%d'),
                    'auc_score': np.nan,
                    'pr_auc': 0.5,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'test_positives': 0,
                })
                continue

            # Train model with best hyperparameters
            lgb_model = lgb.LGBMClassifier(**best_params)
            # params = {'num_threads': 4}
            # lgb_model = lgb.LGBMClassifier(**params)
            lgb_model.fit(X_train, y_train)

            # Predict probabilities
            y_pred_proba = lgb_model.predict_proba(X_test)

            # Calculate AUC and store it
            # We need to make sure there are at least two classes in the test set to calculate AUC
            if True:
                mask = y_test != 2
                auc_score = roc_auc_score(y_test[mask], y_pred_proba[mask][:, 1])

                precision, recall, thresholds = precision_recall_curve(y_test[mask], y_pred_proba[mask][:, 1])
                pr_auc = auc(recall, precision)
                # auc_results.append(auc_score)
                # prauc_results.append(pr_auc)
                results.append({
                    'date': test_date.strftime('%Y-%m-%d'),
                    'auc_score': auc_score,
                    'pr_auc': pr_auc,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'test_positives': y_test[mask].sum()
                })
                print(f"Backtest for {test_date.strftime('%Y-%m')}: AUC = {auc_score:.4f}")
                print(f"Backtest for {test_date.strftime('%Y-%m')}: PR-AUC = {pr_auc:.4f}")
            else:
                print(f"Skipping backtest for {test_date.strftime('%Y-%m')}: Not enough classes in test data.")
        else:
            print(f"Skipping backtest for {test_date.strftime('%Y-%m')}: No test data available.")

    if results:
        results_df = pd.DataFrame(results)
        auc_results = [auc_score for auc_score in auc_results if auc_score is not np.nan]
        mean_auc = np.mean(auc_results)
        mean_prauc = np.mean(prauc_results)
        print(f"\nExpanding Window Backtest Mean AUC: {mean_auc:.4f}")
        return mean_auc, mean_prauc, results_df
    else:
        print("\nNo backtest results were calculated.")
        return None


def plot_backtest_results(results_df, save_path=None):
    """
    Creates comprehensive plots for expanding window backtest results.
    Args:
        results_df (pd.DataFrame): Results from run_expanding_backtest_with_plotting
        save_path (str, optional): Path to save the plot
    """
    if results_df.empty:
        print("No results to plot.")
        return

    # Create a plot with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Expanding Window Backtest Results (Monthly)', fontsize=16, fontweight='bold')

    # Plot 1: AUC Score Over Time
    axes[0].plot(results_df['date'], results_df['auc_score'],
                 marker='o', linewidth=2, markersize=6, color='blue', alpha=0.7)
    axes[0].set_title('AUC Score Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('AUC Score')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: PR-AUC Score Over Time
    axes[1].plot(results_df['date'], results_df['pr_auc'],
                 marker='s', linewidth=2, markersize=6, color='green', alpha=0.7)
    axes[1].set_title('Precision-Recall AUC Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('PR-AUC Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def save_backtesting_results_sql(engine, results_df):
    table_name = 'mlpd_backtesting_results'
    results_df.to_sql(table_name, con=engine, index=False, if_exists="append")
    pass

def main():
    """
    Main function to orchestrate the data loading, splitting, tuning, and training process.
    """
    print("Loading data...")
    root_path = '/data/zhuanghao/MyGithub/MLOps-PDModel/data/input/data'
    sql_conn = MySQLConnection()
    engine = sql_conn._get_engine()
    # df_econ = pd.read_sql("SELECT * FROM mlops_pd.econ;", con=engine)
    # econ_dict = df_econ.set_index('econ_id').to_dict()['econ_name']
    list_econ_data = os.listdir(root_path)

    for econ_file in list_econ_data:

        econ = econ_file.split('_')[2]
        horizon = 1
        df = get_data(path=os.path.join(root_path, econ_file))

        if df is None:
            print("Failed to load data. Exiting.")
            return
        windows_year= 5
        test_year = 2025 - windows_year
        test_month = 6
        expanding_time = f'{windows_year}yrs'

        # Step 1: Prepare the data
        df, features, target_column = prepare_data(df)

        # Step 2: Split the data for hyperparameter tuning and final model training
        splits = split_data(df, features, target_column, test_year=test_year, test_month=test_month)

        # Step 3: Tune hyperparameters using the full training dataset
        # The tuning function will now use TimeSeriesSplit to create expanding windows
        # best_params = tune_hyperparameters(splits['X_train_final'], splits['y_train_final'])

        try:
            with open(f'../../../data/output/mthly_backtesting/tuning_results/best_lgbm_params_{expanding_time}_{econ}_horizon{horizon}.json', 'r') as f:
                best_params = json.load(f)
            print("Loaded best parameters from 'best_lgbm_params.json'.")
        except FileNotFoundError:
            # If the file doesn't exist, run the tuning process
            print("Best parameters file not found. Starting hyperparameter tuning...")
            best_params = tune_hyperparameters(splits['X_train_final'], splits['y_train_final'], expanding_time)
            with open(f'../../../data/output/mthly_backtesting/tuning_results/best_lgbm_params_{expanding_time}_{econ}_horizon{horizon}.json', 'w') as f:
                json.dump(best_params, f, indent=4)

            print(f"\nBest parameters have been saved to 'best_lgbm_params_{expanding_time}_{econ}_horizon{horizon}.json'.")

        best_params['num_threads'] = 4
        # Step 4: Run expanding window backtesting
        # This section gives a more realistic performance estimate over time.
        mean_auc, mean_prauc, results_df = run_expanding_backtest(df, features, target_column, best_params, start_year=test_year, start_month=test_month)
        results_df['econ'] = econ
        results_df['horizon'] = horizon
        results_df['operation_date'] = datetime.now().date()
        save_backtesting_results_sql(engine, results_df)

        # results_df.to_csv(f'/data/zhuanghao/MyGithub/MLOps-PDModel/data/output/mthly_backtesting/mthly_backtest_results_{expanding_time}_{econ}_horizon{horizon}.csv', index=False)
        # Plot results if available
        if not results_df.empty:
            plot_backtest_results(results_df, None)
        print('-'*100)
        print(mean_auc)
        print('-'*100)
        # Step 5: Train final model and make predictions
        # train_and_predict(splits['X_train_final'], splits['y_train_final'],
        #                   splits['X_test'], splits['y_test'],
        #                   best_params)

if __name__ == "__main__":
    main()