from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

class BBGDataSplitter(DataPreprocessingStrategyABC):

    def handle_data(self, df_combined: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:

        self.logger.info('preparing training_pipeline and test data')
        train_df_list = []
        ground_truth_list = []

        # Assuming cols_to_process is defined and contains the columns to be set to np.nan
        # cols_to_process = ['col1', 'col2', ...]
        for id_bb_unique, sub_df in tqdm(df_combined.groupby('ID_BB_UNIQUE'),
                                         desc="Splitting data into train and test"):

            # if deploy:
            train_sub_df = sub_df[sub_df.Year.apply(lambda x: x <= 2020)]
            test_sub_df = sub_df[sub_df.Year.apply(lambda x: x > 2020)]
            # else:
            #     # Calculate the number of rows to include (first 80%)
            #     n_rows = len(sub_df)
            #     n_train = int(n_rows * 0.8)
            #
            #     # # Split the data into training_pipeline and testing subsets
            #     train_sub_df = sub_df.iloc[:n_train]
            #     test_sub_df = sub_df.iloc[n_train:]

            train_df_list.append(train_sub_df.copy())
            # Store ground truth for the test set
            ground_truth_list.append(test_sub_df.copy())

            # # Set specified columns in test_sub_df to np.nan
            # test_sub_df[self.x_cols_to_process] = np.nan
            #
            # # # Step 1: Drop columns with high missing values marked as np.nan
            # # df = self.bbgdatamvhandler.mark_high_missing_columns(train_sub_df, threshold=0.8)
            # #
            # # # Step 2: Handle missing values for each id_bb_unique
            # # df_processed = self.bbgdatamvhandler.impute_missing_values(df)
            #
            # # Combine processed training_pipeline data with modified test data
            # df_train = pd.concat([df_processed, test_sub_df], ignore_index=True)
            #
            # # Add the combined data to the train list
            # train_df_list.append(df_train)

        return train_df_list, ground_truth_list