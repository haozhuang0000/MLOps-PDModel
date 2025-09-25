from datetime import datetime
import sys
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))  # Adjust if needed

# Add the root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)


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
from src.mlops.database.db_connection import MySQLConnection
from src.modeldev.model.lgbm_classifier_yearly import get_data, prepare_data, run_expanding_backtest_with_shap, split_data, plot_backtest_results, tune_hyperparameters
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


from sqlalchemy import Table, MetaData
from sqlalchemy.dialects.mysql import insert
import numpy as np

def save_backtesting_results_sql(engine, results_df):
    table_name = 'mlpd_backtesting_results'
    
    # Replace NaN/NaT with None (so MySQL gets NULL)
    clean_df = results_df.replace({np.nan: None})

    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]

    with engine.begin() as conn:
        for row in clean_df.to_dict(orient='records'):
            stmt = insert(table).values(**row).prefix_with("IGNORE")
            conn.execute(stmt)

import os
import glob
import json
import pandas as pd


from sqlalchemy import text  # Add this import at the top of your script

def check_econ_exists_in_db(engine, econ):
    query = text(f"SELECT COUNT(*) FROM mlpd_backtesting_results WHERE econ = :econ AND horizon = 12")
    with engine.begin() as connection:
        result = connection.execute(query, {"econ": econ})
        count = result.scalar()
    return count > 0




def main():
    """
    Main function to orchestrate the data loading, splitting, tuning, and training process.
    Modified to handle multiple CSV files from a folder.
    """

    input_folder = "/data 1/data/"
    file_pattern = "merged_output_*_202508.csv"
    csv_files = glob.glob(os.path.join(input_folder, file_pattern))

    sql_conn = MySQLConnection()
    engine = sql_conn._get_engine()

    if not csv_files:
        print(f"No CSV files found in {input_folder} matching pattern {file_pattern}.")
        return

    # Output folders
    backtesting_folder = 'lightgbm_results/backtesting/'
    plots_folder = 'lightgbm_results/plots/'
    shap_plots_folder = 'lightgbm_results/shap_plots/'
    metrics_folder = 'lightgbm_metrics/'

    # Create directories if they don't exist
    os.makedirs(backtesting_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(shap_plots_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)

    for file_path in csv_files:
        print(f"\nProcessing file: {file_path}")

        # Extract econ from filename: merged_output_{econ}_202508.csv
        filename = os.path.basename(file_path)
        try:
            econ = filename.split('_')[2]  # Gets the part between output_ and _202508
        except IndexError:
            print(f"Failed to extract econ from filename: {filename}. Skipping.")
            continue

        # Paths for checking existence
        params_filename = f'best_lgbm_params_12M_{econ}.json'
        params_path = os.path.join(backtesting_folder, params_filename)
        metrics_file = os.path.join(metrics_folder, f'12M_backtest_results_{econ}.csv')

        # Check if already inserted in MySQL
        is_in_db = check_econ_exists_in_db(engine, econ)

        # Skip if best params, metrics file, and MySQL entry all exist
        if os.path.exists(params_path) and os.path.exists(metrics_file) and is_in_db:
            print(f"Skipping {econ}: Parameters, backtest results, and DB record already exist.")
            continue
        
        #df = pd.read_csv(file_path)
        print("Loading data...")
        df = get_data(file_path)

        if df is None or df.empty:
            print(f"Failed to load or empty data in {file_path}. Skipping.")
            continue

        print("ECON",econ)

        # Configuration
        windows_years = 10
        test_year = 2025 - windows_years
        expanding_time = f'{windows_years}yrs'

        # Step 1: Prepare the data
        df, features, target_column = prepare_data(df)

        # Step 2: Split the data
        splits = split_data(df, features, target_column, test_year=test_year)

        # Step 3: Load or tune hyperparameters
        params_filename = f'best_lgbm_params_12M_{econ}.json'
        params_path = os.path.join(backtesting_folder, params_filename)

        try:
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            print("Loaded best parameters from file.")
        except FileNotFoundError:
            print("Best parameters not found. Running tuning...")
            os.makedirs(os.path.dirname(params_path), exist_ok=True)  # Ensure folder exists
            best_params = tune_hyperparameters(
                splits['X_train_final'], splits['y_train_final'], f'yearly_{expanding_time}_{econ}'
            )
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"Saved best parameters to {params_path}")
        

        best_params['num_threads'] = 4

        shap_path = os.path.join(shap_plots_folder, f'12M_backtest_shap_{econ}.png')

        # Step 4: Backtesting with SHAP
        auc_mean, prauc_mean, results_df, shap_importance_df = run_expanding_backtest_with_shap(
            df, features, target_column, best_params, start_year=2015, shap_summary_path='./', plots_path=shap_path
        )

        # Save metrics CSV
        metrics_file = os.path.join(metrics_folder, f'12M_backtest_results_{econ}.csv')
        results_df.to_csv(metrics_file, index=False)

        # Save plot
        if not results_df.empty:
            plot_file = os.path.join(plots_folder, f'12M_backtest_results_{econ}.png')
            plot_backtest_results(results_df, plot_file)

        results_df.rename(columns={'year': 'date'}, inplace=True)
        if 'test_samples' in results_df.columns:
            results_df.drop(columns='test_samples', inplace=True)
        else:
            print("'test_samples' column not found in results_df. Skipping drop.")

        results_df['econ'] = econ
        results_df['horizon'] = 12
        results_df['operation_date'] = datetime.now().date()
        save_backtesting_results_sql(engine, results_df)

        print('-' * 80)
        if auc_mean is not None and prauc_mean is not None:
            print(f"[{econ}] AUC: {auc_mean:.4f}, PR-AUC: {prauc_mean:.4f}")
        else:
            print(f"[{econ}] AUC or PR-AUC not available (likely due to skipped test windows).")
        print('-' * 80)

        

if __name__ == "__main__":
    main()