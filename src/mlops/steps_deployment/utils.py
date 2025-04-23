import pandas as pd

# Load the dataset
file_path = '/path/to/your/union_processed.csv'
data = pd.read_csv(file_path)

# Define the categorical and numerical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
numerical_columns = data.select_dtypes(include=['float64']).columns.tolist()
primary_key_columns = ['Year']

# Ensure primary key columns are not missed
categorical_columns = [col for col in categorical_columns if col not in primary_key_columns]


# Function to extend the data
def extend_dates(group, start_year, end_year):
    """Extend the Year column for each group and fill with NaNs for numerical columns."""
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


# Extend the dataset from the minimum year to 2030
start_year = data['Year'].min()
end_year = 2030
extended_data = data.groupby('ID_BB_UNIQUE', group_keys=False).apply(
    lambda group: extend_dates(group, start_year, end_year)
).reset_index(drop=True)

# Save the extended dataset
output_file = '/path/to/save/union_extended_to_2030.csv'
extended_data.to_csv(output_file, index=False)

print(f"Dataset successfully extended and saved to: {output_file}")
