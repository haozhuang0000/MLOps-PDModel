from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union
import pandas as pd

class BBGDataMerger(DataPreprocessingStrategyABC):

    def handle_data(self,
                    df_annual_sorted_after_2000: pd.DataFrame,
                    industry_mappings: list,
                    df_company_info: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, list]:

        self.logger.info('XY and Industry -> merged data')
        industry_cols_to_merge = []  # List to store names of the new mapped columns

        # Iterate over each column and its corresponding mapping
        for col, mapping in zip(self.industry_cols, industry_mappings):
            new_col_name = col + '_mapped'  # Generate new column name
            industry_cols_to_merge.append(new_col_name)  # Add new column name to the list

            # Create the new column by mapping values using the provided dictionary
            df_company_info[new_col_name] = df_company_info[col].map(mapping)

        # Combine the industry mapped columns and the 'ID_BB_UNIQUE' column into a list
        columns_to_select = industry_cols_to_merge + ['ID_BB_UNIQUE']

        # Perform a left join to merge the two DataFrames on the 'ID_BB_UNIQUE' column
        df_merged = pd.merge(
            df_annual_sorted_after_2000,
            df_company_info[columns_to_select],
            on='ID_BB_UNIQUE',
            how='left'
        )

        # Display the merged DataFrame
        return df_merged, industry_cols_to_merge