from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class BBGDataUnionhelper(DataPreprocessingStrategyABC):

    def handle_data(self,
                    x: pd.Series,
                    update_dt_col: pd.Series,
                    eqy_consolidated_col: pd.Series,
                    filing_status_col: pd.Series,
                    accounting_standard_col: pd.Series) -> Union[float, np.nan]:

        """
                Aggregates values in a column by applying prioritization rules to resolve cases with multiple unique values.

                Parameters:
                    x (Pandas Series): The main column for aggregation, containing raw data.
                    update_dt_col (Pandas Series): A column with update timestamps to prioritize the latest rows.
                    eqy_consolidated_col (Pandas Series): A column indicating if a record is consolidated ('Y' or 'N').
                    filing_status_col (Pandas Series): A column with filing status values prioritized as follows:
                        - 'MR' (Most Recent): Priority 1
                        - 'OR': Priority 2
                        - 'PR': Priority 3
                        - 'RS': Priority 4
                        - 'ER': Priority 5
                    accounting_standard_col (Pandas Series): A column indicating the accounting standard prioritized as follows:
                        - 'IAS/IFRS': Priority 1
                        - 'US GAAP': Priority 2

                Returns:
                    Single value (any type): The resolved value from column `x`.
                    np.nan: If no resolution can be achieved or if `x` is empty after dropping NaN values.
                """

        # Define the priority mapping for 'FILING_STATUS'
        filing_status_priority = self.filing_status_priority

        # Define the priority mapping for 'ACCOUNTING_STANDARD'
        accounting_standard_priority = self.accounting_standard_priority

        x = x.dropna()
        if x.empty:
            return np.nan

        unique_values = x.unique()

        if len(unique_values) == 1:
            return unique_values[0]

        # Rule-Based Selection of Value: When there are more than one unique_values we apply the rules to filter out the correct value.
        # Step 1: Try to resolve by latest FUNDAMENTAL_UPDATE_DT
        dt_values = update_dt_col.loc[x.index]
        max_date = dt_values.max()
        latest_mask = dt_values == max_date
        latest_indices = x.index[latest_mask]
        latest_values = x.loc[latest_indices]
        unique_latest_values = latest_values.unique()

        if len(unique_latest_values) == 1:
            return unique_latest_values[0]

        # Step 2: If undecided, prefer EQY_CONSOLIDATED == 'Y'
        eqy_values = eqy_consolidated_col.loc[latest_indices]
        y_mask = eqy_values == 'Y'
        if y_mask.any():
            y_indices = latest_indices[y_mask]
            y_values = x.loc[y_indices]
            unique_y_values = y_values.unique()
            if len(unique_y_values) == 1:
                return unique_y_values[0]
            else:
                # Step 3: If still undecided, use FILING_STATUS priority
                filing_status_values = filing_status_col.loc[y_indices]
                priorities = filing_status_values.map(filing_status_priority)
                min_priority = priorities.min()
                priority_mask = priorities == min_priority
                priority_indices = y_indices[priority_mask]
                priority_values = x.loc[priority_indices]
                unique_priority_values = priority_values.unique()
                if len(unique_priority_values) == 1:
                    return unique_priority_values[0]
                else:
                    # Step 4: If still undecided, use ACCOUNTING_STANDARD priority
                    accounting_standard_values = accounting_standard_col.loc[priority_indices]
                    acc_priorities = accounting_standard_values.map(accounting_standard_priority)
                    min_acc_priority = acc_priorities.min()
                    acc_priority_mask = acc_priorities == min_acc_priority
                    acc_priority_indices = priority_indices[acc_priority_mask]
                    acc_priority_values = x.loc[acc_priority_indices]
                    unique_acc_priority_values = acc_priority_values.unique()
                    if len(unique_acc_priority_values) == 1:
                        return unique_acc_priority_values[0]
                    else:
                        return np.nan
        else:
            # If no EQY_CONSOLIDATED == 'Y', proceed to FILING_STATUS
            filing_status_values = filing_status_col.loc[latest_indices]
            priorities = filing_status_values.map(filing_status_priority)
            min_priority = priorities.min()
            priority_mask = priorities == min_priority
            priority_indices = latest_indices[priority_mask]
            priority_values = x.loc[priority_indices]
            unique_priority_values = priority_values.unique()
            if len(unique_priority_values) == 1:
                return unique_priority_values[0]
            else:
                # Step 4: If still undecided, use ACCOUNTING_STANDARD priority
                accounting_standard_values = accounting_standard_col.loc[priority_indices]
                acc_priorities = accounting_standard_values.map(accounting_standard_priority)
                min_acc_priority = acc_priorities.min()
                acc_priority_mask = acc_priorities == min_acc_priority
                acc_priority_indices = priority_indices[acc_priority_mask]
                acc_priority_values = x.loc[acc_priority_indices]
                unique_acc_priority_values = acc_priority_values.unique()
                if len(unique_acc_priority_values) == 1:
                    return unique_acc_priority_values[0]
                else:
                    return np.nan

class BBGDataUnion(DataPreprocessingStrategyABC):

    def handle_data(self,
                    df_merged: pd.DataFrame,
                    industry_cols_to_merge: list) -> Union[pd.DataFrame, pd.Series, list]:
        # Initialize an empty list to store the result DataFrames for each ID_BB_UNIQUE
        result = []
        self.logger.info('applying BBGDataAggregation.custom_agg to keep only one unique value per company per date'
                         '(update_date, consolidated, filing_status, accounting_standard)')
        # Combine columns to process from x_cols_to_process and industry_cols_to_merge
        cols_to_process = self.x_cols_to_process + industry_cols_to_merge

        # Group the merged DataFrame by 'ID_BB_UNIQUE' and process each group
        for id_bb_unique, sub_df in tqdm(df_merged.groupby('ID_BB_UNIQUE'),
                                         desc="Processing ID_BB_UNIQUE groups - bbg_data_aggregation.py - [union_processing]"):
            # Process each group of rows for the current TICKER, grouped further by 'Year'
            df_result = sub_df.groupby('Year').apply(
                lambda year: pd.Series(
                    {
                        'Year': year.name,  # Extract the 'Year' as the index name
                        # Apply the custom aggregation function to each column in cols_to_process
                        **{
                            col: BBGDataUnionhelper().handle_data(
                                year[col],  # Column data for processing
                                year['FUNDAMENTAL_UPDATE_DT'],  # Update date information
                                year['EQY_CONSOLIDATED'],  # Consolidation status
                                year['FILING_STATUS'],  # Filing status
                                year['ACCOUNTING_STANDARD']  # Accounting standard
                            )
                            for col in cols_to_process
                        }
                    }
                )
            ).reset_index(drop=True)  # Flatten the result by resetting the index

            # Insert the 'ID_BB_UNIQUE' column to associate the processed data with the corresponding id_bb_unique
            df_result.insert(1, 'ID_BB_UNIQUE', list(sub_df['ID_BB_UNIQUE'])[0])  # Use the first ID_BB_UNIQUE value in the sub-group

            # Append the processed DataFrame for this ID_BB_UNIQUE to the result list
            result.append(df_result)

        df_combined = pd.concat(result, ignore_index=True)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output'))
        df_combined.to_csv(os.path.join(base_dir, 'union_processed.csv'), index=False)
        return df_combined, cols_to_process