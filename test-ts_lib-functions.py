import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
from ts_lib import plot_ts  
from ts_lib import append_datetime_cols
from ts_lib import periodic_kde
from ts_lib import create_windowed_df  
from ts_lib import create_windowed_df
from unittest.mock import patch, MagicMock
from ts_lib import seasonal_decompositions  
from ts_lib import seasonal_catplot
from statsmodels.tsa.seasonal import seasonal_decompose


#####################################################################################################################

# Test the plot_ts function

def test_plot_ts():
    # Create a sample DataFrame
    data = {'date': pd.date_range(start='1/1/2020', periods=5),
            'value': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    # Mock plt.show() to not actually display the plot during the test
    with patch('matplotlib.pyplot.show') as mocked_show:
        plot_ts(df, 'value')
        # Verify plt.show() was called once
        mocked_show.assert_called_once()

#####################################################################################################################

#Tests for append_datetime_cols

def test_append_datetime_cols_with_column():
    # DataFrame without datetime index but with a datetime column
    data = {'date': pd.date_range(start='2020-01-01', periods=3, freq='D'),
            'value': [1, 2, 3]}
    df = pd.DataFrame(data)

    # Apply function
    result_df = append_datetime_cols(df, col='date')

    # Check new columns are added
    assert all(col in result_df.columns for col in ['year', 'month', 'day', 'week_of_year', 'week_of_month']), "Not all datetime components were added correctly"

def test_append_datetime_cols_with_index():
    # DataFrame with datetime index
    data = {'value': [1, 2, 3]}
    dates = pd.date_range(start='2020-01-01', periods=3, freq='D')
    df = pd.DataFrame(data, index=dates)

    # Apply function with index=True
    result_df = append_datetime_cols(df, index=True)

    # Check new columns are added
    assert all(col in result_df.columns for col in ['year', 'month', 'day', 'week_of_year', 'week_of_month']), "Datetime components not added correctly for datetime indexed DataFrame"

def test_append_datetime_cols_format_parsing():
    # DataFrame with non-standard datetime format
    data = {'date': ['01-2020-01', '02-2020-01', '03-2020-01'],
            'value': [1, 2, 3]}
    df = pd.DataFrame(data)

    # Apply function with a custom datetime format
    result_df = append_datetime_cols(df, col='date', dt_format='%d-%Y-%m')

    # Check if dates were parsed correctly
    assert all(result_df['year'] == 2020), "Year was not parsed correctly"
    assert all(result_df['month'] == 1), "Month was not parsed correctly"

#####################################################################################################################

# Test the periodic_kde_plot function
def test_periodic_kde_plot_grouping():
    # Sample DataFrame with monthly data
    data = {'date': pd.date_range(start='2020-01-01', periods=12, freq='M'),
            'value': range(1, 13)}
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    # Mock plt.show to not display the plot, assuming your function uses plt.show() to display each plot
    with patch('matplotlib.pyplot.show') as mocked_show:
        # Execute the function with the DataFrame
        periodic_kde(df, 'value', 'M', index=True)
        
        # Verify plt.show() was called 12 times, once for each month with data
        assert mocked_show.call_count == 12, "Plot was not called once for each month with data"


#####################################################################################################################

# Test the create_windowed_df function


def test_create_windowed_df():
    # Sample time series DataFrame
    data = {'value': range(10)}
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    df = pd.DataFrame(data, index=dates)

    # Apply function
    num_periods = 3
    result_df = create_windowed_df(df, num_periods, 'value')

    # Check the shape of the resulting DataFrame
    assert result_df.shape == (7, 4), "Windowed DataFrame shape is incorrect"
    # Check if the correct columns were created
    expected_columns = ['value_t1', 'value_t2', 'value_t3', 'value_t4']
    assert all(col in result_df.columns for col in expected_columns), "Not all expected shifted columns were created"

#####################################################################################################################
# Test the seasonal_decompositions function

def test_seasonal_decompositions():
    data = {'date': pd.date_range(start='2020-01-01', periods=365, freq='D'),
            'value': np.random.rand(365)}
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    with patch('statsmodels.tsa.seasonal.seasonal_decompose') as mock_decompose, \
         patch('matplotlib.pyplot.show'):
        # Call the function under test
        seasonal_decompositions(df, 'value')
        # Now use mock_decompose to check if it was called
        mock_decompose.assert_called()

#####################################################################################################################
# Test the seasonal_catplot function

# Test for seasonal_catplot with boxplot
def test_seasonal_catplot_boxplot():
    data = {'date': pd.date_range(start='2020-01-01', periods=365, freq='D'),
            'USD': np.random.rand(365)}
    df = pd.DataFrame(data)

    with patch('seaborn.boxplot') as mocked_boxplot, \
         patch('matplotlib.pyplot.show'):
        seasonal_catplot(df, 'USD', 'boxplot', col='date')
        # Check if seaborn.boxplot is called at least once
        mocked_boxplot.assert_called()

# Test for seasonal_catplot with violinplot
def test_seasonal_catplot_violinplot():
    data = {'date': pd.date_range(start='2020-01-01', periods=365, freq='D'),
            'USD': np.random.rand(365)}
    df = pd.DataFrame(data)

    with patch('seaborn.violinplot') as mocked_violinplot, \
         patch('matplotlib.pyplot.show'):
        seasonal_catplot(df, 'USD', 'violinplot', col='date')
        # Check if seaborn.violinplot is called at least once
        mocked_violinplot.assert_called()