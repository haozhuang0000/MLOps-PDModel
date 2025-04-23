from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union, Tuple, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

class PDDataSplitter(DataPreprocessingStrategyABC):

    @staticmethod
    def create_time_splits(df, n_splits=5, test_years=3):
        """
        Uses sklearn's TimeSeriesSplit to generate train/val sets based on row order,
        and holds out the most recent years as a test set.

        Assumes the DataFrame has 'yyyy' and 'mm' columns and creates a full date column.

        Args:
            df: Full DataFrame with 'yyyy' and 'mm' columns.
            n_splits: Number of CV splits.
            test_years: Number of years to hold out for testing.

        Returns:
            train_val_splits: List of dicts with 'train' and 'val' DataFrames.
            test_splits: Final test set DataFrame.
        """
        # Create full datetime column
        df['date'] = pd.to_datetime(df['yyyy'].astype(str) + '-' + df['mm'].astype(str).str.zfill(2) + '-01')
        df = df.sort_values(by='date').reset_index(drop=True)

        # Define which years go into the test set
        all_years = df['yyyy'].unique()
        final_test_years = all_years[-test_years:]
        train_val_df = df[~df['yyyy'].isin(final_test_years)]

        # TimeSeriesSplit works on index (row-wise)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        train_val_splits = []

        for i, (train_idx, val_idx) in enumerate(tscv.split(train_val_df)):
            print(f'Fold {i + 1}: ')
            print(f"Train rows {train_idx[0]}–{train_idx[-1]}, Val rows {val_idx[0]}–{val_idx[-1]}")
            print("Train period:", df.iloc[train_idx]['date'].min(), "to", df.iloc[train_idx]['date'].max())
            print("Val period:", df.iloc[val_idx]['date'].min(), "to", df.iloc[val_idx]['date'].max())
            print("-" * 100)
            split = {
                "train": train_val_df.iloc[train_idx],
                "val": train_val_df.iloc[val_idx]
            }
            train_val_splits.append(split)

        # Final test set
        test_splits = df[df['yyyy'].isin(final_test_years)]

        return train_val_splits, train_val_df, test_splits

    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, List]:

        # X = df.drop(columns=["Y"])
        # y = df["Y"]
        # # # Step 1: First, split the data into train (60%) and temp (40%) [which will be further split into validation and test]
        # # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y, shuffle=True)
        # #
        # # # Step 2: Split temp into validation (20%) and test (20%)
        # # X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,
        # #                                                     stratify=y_temp, shuffle=True)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)
        # X_valid = X_test.copy()
        # y_valid = y_test.copy()
        # return X_train, X_valid, X_test, y_train, y_valid, y_test
        train_val_splits, train_df, final_test = self.create_time_splits(df)

        return train_val_splits, train_df, final_test