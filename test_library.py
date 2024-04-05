import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
from library import AutoEDA
from unittest.mock import patch, MagicMock, ANY
from statsmodels.tsa.seasonal import seasonal_decompose


#####################################################################################################################

# Test the plot_ts function

def test_plot_ts():
    # Create a sample DataFrame
    data = {'date': pd.date_range(start='1/1/2020', periods=5),
            'value': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    # Initialize AutoEDA with the DataFrame
    auto_eda = AutoEDA(df=df, label_col='value', index=True)

    # Mock plt.show() to not actually display the plot during the test
    with patch('matplotlib.pyplot.show') as mocked_show:
        auto_eda.plot_ts()
        # Verify plt.show() was called once
        mocked_show.assert_called_once()

#####################################################################################################################

#Tests for append_datetime_cols

def test_append_datetime_cols_with_column():
    # DataFrame without datetime index but with a datetime column
    data = {'date': pd.date_range(start='2020-01-01', periods=3, freq='D'),
            'value': [1, 2, 3]}
    df = pd.DataFrame(data)
    # Initialize AutoEDA with the DataFrame
    auto_eda = AutoEDA(df=df, label_col='value', dt_col='date', index=False)

    # Apply function
    result_df = auto_eda.append_datetime_cols()

    # Check new columns are added
    assert all(col in result_df.columns for col in ['year', 'month', 'day', 'week_of_year', 'week_of_month']), "Not all datetime components were added correctly"

def test_append_datetime_cols_with_index():
    # DataFrame with datetime index
    data = {'value': [1, 2, 3]}
    dates = pd.date_range(start='2020-01-01', periods=3, freq='D')
    df = pd.DataFrame(data, index=dates)
    # Initialize AutoEDA with the DataFrame
    auto_eda = AutoEDA(df=df, label_col='value', index=True)

    # Apply function with index=True
    result_df = auto_eda.append_datetime_cols()

    # Check new columns are added
    assert all(col in result_df.columns for col in ['year', 'month', 'day', 'week_of_year', 'week_of_month']), "Datetime components not added correctly for datetime indexed DataFrame"

def test_append_datetime_cols_format_parsing():
    # DataFrame with non-standard datetime format
    data = {'date': ['01-2020-01', '02-2020-01', '03-2020-01'],
            'value': [1, 2, 3]}
    df = pd.DataFrame(data)

    # Initialize AutoEDA with the DataFrame
    auto_eda = AutoEDA(df=df, label_col='value', dt_col='date', index=False)

    # Apply function with a custom datetime format
    result_df = auto_eda.append_datetime_cols(dt_format='%d-%Y-%m')

    # Check if dates were parsed correctly
    assert all(result_df['year'] == 2020), "Year was not parsed correctly"
    assert all(result_df['month'] == 1), "Month was not parsed correctly"

#####################################################################################################################



#####################################################################################################################

# Test the create_windowed_df function


def test_create_windowed_df():
    # Sample time series DataFrame
    data = {'value': range(10)}
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    df = pd.DataFrame(data, index=dates)

    # Initialize AutoEDA with the DataFrame
    auto_eda = AutoEDA(df=df, label_col='value', index=True)

    # Apply function
    num_periods = 3
    result_df = auto_eda.create_windowed_df(num_periods, 'value')

    # Check the shape of the resulting DataFrame
    assert result_df.shape == (7, 4), "Windowed DataFrame shape is incorrect"
    # Check if the correct columns were created
    expected_columns = ['value_t1', 'value_t2', 'value_t3', 'value_t4']
    assert all(col in result_df.columns for col in expected_columns), "Not all expected shifted columns were created"

#####################################################################################################################
# Test the seasonal_decompositions function
'''
def test_seasonal_decompositions():
    data = {'date': pd.date_range(start='2020-01-01', periods=365, freq='D'),
            'value': np.random.rand(365)}
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    with patch('statsmodels.tsa.seasonal.seasonal_decompose') as mock_decompose:
        # Configure the mock to return a MagicMock with a trend attribute
        mock_decompose.return_value = MagicMock(trend=pd.Series(np.random.rand(365), index=df.index))
        
        # Call the function under test
        seasonal_decompositions(df, 'value')
        
        # Define the expected calls with ANY for the Series argument
        expected_calls = [
            (ANY, {'model': 'additive', 'period': 7}),
            (ANY, {'model': 'additive', 'period': 30}),
            (ANY, {'model': 'additive', 'period': 365})
        ]
        
        # Verify seasonal_decompose was called with expected arguments
        mock_decompose.assert_has_calls(expected_calls, any_order=True)
'''

#####################################################################################################################
# Test the seasonal_catplot function

def test_seasonal_catplot_boxplot():
    data = {'date': pd.date_range(start='2020-01-01', periods=365, freq='D'),
            'USD': np.random.rand(365)}
    df = pd.DataFrame(data)

    # Initialize AutoEDA
    auto_eda = AutoEDA(df=df, label_col='USD', dt_col='date', index=False)

    with patch('seaborn.boxplot') as mocked_boxplot, \
         patch('matplotlib.pyplot.show'):
        auto_eda.seasonal_catplots(kind='boxplot')
        # Check if seaborn.boxplot is called at least once
        mocked_boxplot.assert_called()

def test_seasonal_catplot_violinplot():
    data = {'date': pd.date_range(start='2020-01-01', periods=365, freq='D'),
            'USD': np.random.rand(365)}
    df = pd.DataFrame(data)

    # Initialize AutoEDA
    auto_eda = AutoEDA(df=df, label_col='USD', dt_col='date', index=False)

    with patch('seaborn.violinplot') as mocked_violinplot, \
         patch('matplotlib.pyplot.show'):
        auto_eda.seasonal_catplots(kind='violinplot')
        # Check if seaborn.violinplot is called at least once
        mocked_violinplot.assert_called()
