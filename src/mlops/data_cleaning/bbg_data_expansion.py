from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union
import pandas as pd

# Function to extend the data
def extend_dates(group, start_year, end_year, categorical_columns, numerical_columns):
    """Extend the Year column for each group and fill with NaNs for numerical columns."""

    start_year = int(start_year)
    end_year = int(end_year)

    full_years = pd.DataFrame({'Year': range(start_year, end_year + 1)})
    extended_group = pd.merge(full_years, group, on='Year', how='left')

    # Fill categorical columns with the existing values (forward fill)
    for cat_col in categorical_columns:
        extended_group[cat_col] = extended_group[cat_col].fillna(method='ffill').fillna(method='bfill')

    # Ensure numerical columns are NaN for new rows
    extended_group[numerical_columns] = extended_group[numerical_columns].where(
        ~extended_group[numerical_columns].isna(), other=pd.NA
    )
    return extended_group

class BBGDataExpansion(DataPreprocessingStrategyABC):

    def handle_data(self, df_combined: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, list, dict]:

        # Define the categorical and numerical columns
        categorical_columns = df_combined.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = df_combined.select_dtypes(include=['float64']).columns.tolist()
        expand_key_columns = ['Year']

        # Ensure primary key columns are not missed
        categorical_columns = [col for col in categorical_columns if col not in expand_key_columns]

        # Extend the dataset from the minimum year to 2030
        start_year = df_combined['Year'].min()
        end_year = self.expand_year
        extended_data = df_combined.groupby('ID_BB_UNIQUE', group_keys=False).apply(
            lambda group: extend_dates(group, start_year, end_year, categorical_columns, numerical_columns)
        ).reset_index(drop=True)
        extended_data.to_csv('../../../data/output/union_processed_expanded.csv', index=False)
        return extended_data


