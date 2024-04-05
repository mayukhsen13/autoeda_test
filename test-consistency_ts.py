import pandas as pd
import numpy as np
from example_timeseries import df

'''
def check_column_homogeneity(dataframe):
    for column_name, column_data in dataframe.iteritems():
        # Infer the column data type based on the first non-NA value
        inferred_column_dtype = pd.api.types.infer_dtype(column_data.dropna(), skipna=True)
        
        # Check if the column contains mixed types
        if inferred_column_dtype == "mixed":
            raise ValueError(f"Column '{column_name}' contains mixed data types.")
        else:
            print(f"Column '{column_name}' is homogeneous ({inferred_column_dtype}).")
           
 '''


# Function to check for missing (NA) values
def check_missing_values(dataframe):
    if dataframe.isnull().sum().sum() > 0:
        raise ValueError("Data contains NA or missing values.")
    else:
        print("No missing values detected.")

# Function to check for duplicated rows
def check_duplicates(dataframe):
    if dataframe.duplicated().sum() > 0:
        raise ValueError("Data contains duplicated rows.")
    else:
        print("No duplicated rows detected.")

'''
# Function to check for consistency in datetime columns
def check_datetime_consistency(dataframe):
    # Check if there are any datetime columns in the DataFrame
    datetime_columns = dataframe.select_dtypes(include='datetime64').columns
    if len(datetime_columns) == 0:
        raise ValueError("No datetime columns found in the DataFrame.")
    
    # Check if the datetime columns are consistent
    for col in datetime_columns:
        if not dataframe[col].is_monotonic_increasing:
            raise ValueError(f"Values in datetime column '{col}' are not in increasing order.")
        if not dataframe[col].is_unique:
            raise ValueError(f"Values in datetime column '{col}' are not unique.")
    
    print("Datetime columns are consistent.")
'''

# Function to validate the DataFrame schema
def validate_schema(dataframe, expected_columns_and_types):
    for column, expected_type in expected_columns_and_types.items():
        if column not in dataframe.columns:
            raise ValueError(f"Missing expected column: {column}")
        actual_type = dataframe[column].dtype
        if not np.issubdtype(actual_type, expected_type):
            raise ValueError(f"Incorrect data type for {column}. Expected {expected_type}, got {actual_type}.")
        
    print("Schema validation succeeded: All columns are present with correct data types.")
    
# Main function to run data quality checks
def run_data_quality_checks(dataframe):
    expected_columns_and_types = {
        # Replace with your actual expected columns and their data types
        'USD': np.float64,
        'JPY': np.float64,
        'BGN': np.float64,
        'CZK': np.float64,
        'DKK': np.float64,
        'GBP': np.float64,
        'CHF': np.float64,
        # Add more columns as necessary
    }
    print("Running data quality checks...")
    check_missing_values(dataframe)
    check_duplicates(dataframe)
    #check_column_homogeneity(dataframe)
    #check_datetime_consistency(dataframe)
    validate_schema(dataframe, expected_columns_and_types)
    print("All data quality checks passed successfully.")

run_data_quality_checks(df)

'''
csv_file_path = 'df_export.csv'
df = pd.read_csv(csv_file_path)
'''
