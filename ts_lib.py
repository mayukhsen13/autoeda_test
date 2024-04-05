import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib as mpl
import calendar


#####################################################################################################################

### Visualization ###


def plot_ts(df:pd.DataFrame, label_col:str): 
    '''
    Returns time series lineplot from univariate time series variable 

    : df : pd.DataFrame with 'datetime64[ns]' index and single time series variable
    : label_col : variable of interest
    '''
    df[label_col].plot(figsize=(20,10), title=label_col)
    plt.show()


def append_datetime_cols(df:pd.DataFrame, col:str='none', dt_format:str='%Y-%m-%d', index=False, return_dt_index=False) -> pd.DataFrame:
    
    '''
    Extracts datetime components from daily datetime column of a pandas DataFrame.
    If DataFrame has no datetime column but is datetime indexed, index=True
    Returns original pandas DataFrame augmented with datetime components.

    : df : DataFrame containing datetime column with dt_format string format
    : col : specified datetime indexed column with dt_format string format
    : dt_format : string format for col
    : index : boolean, true if Dataframe is datetime indexec (and no datetime column)
    : return_dt_index : boolean, if original datetime column (or index) is wished to be returned as the DataFrame index
    '''
    
    if index == True:
        t_df = df.reset_index().rename(columns={'index':'date'})
        col = 'date'
    else:
        t_df = df.copy()

    dt_col = pd.to_datetime(t_df[col],
                            format=dt_format,
                            errors='coerce')
    t_df[col] = dt_col
    t_df['year'], t_df['month'], t_df['day'] = dt_col.dt.year, dt_col.dt.month, dt_col.dt.day
    t_df['week_of_year'] = dt_col.dt.isocalendar().week
    t_df['week_of_month'] = dt_col.apply(lambda d: (d.day-1) // 7 + 1)
    t_df['day_of_week'] = dt_col.dt.dayofweek.apply(lambda d: d+1)

    if return_dt_index == True:
        t_df = t_df.set_index('date')
        t_df.index.name = None

    return t_df


def seasonal_catplot(df, label_col, kind, col:str='none', dt_format:str='%Y-%m-%d', index=False):
    ''' 
    Takes univariate daily time series pandas DataFrame and returns seasonal datetime categorical plots

    : df : daily time series pandas DataFrame
    : label_col : target variable
    : kind : either 'boxplot' or 'violinplot' 
    : col : include if datetime information is a variable and not the index
    : dt_format : datetime format
    : index : boolean, true if datetime information is the DataFrame index
    '''

    dates_df = append_datetime_cols(df, col, dt_format, index, return_dt_index=False)

    if kind=='boxplot':
        fig, axs = plt.subplots(5, 1, figsize=(20,30))
        sns.boxplot(dates_df, x='day_of_week', y='USD', ax=axs[0])
        axs[0].set_title(f'{label_col} by Day of Week')

        sns.boxplot(dates_df, x='month', y='USD', ax=axs[1])
        axs[1].set_title(f'{label_col} by Month')

        sns.boxplot(dates_df, x='week_of_month', y='USD', ax=axs[2])
        axs[2].set_title(f'{label_col} by Weeek of Month')

        sns.boxplot(dates_df, x='week_of_year', y='USD', ax=axs[3])
        axs[3].set_title(f'{label_col} by Weeek of Year')

        sns.boxplot(dates_df, x='year', y='USD', ax=axs[4])
        axs[4].set_title(f'{label_col} by Year')

        plt.show()

    if kind=='violinplot':
        fig, axs = plt.subplots(5, 1, figsize=(20,30))
        sns.violinplot(dates_df, x='day_of_week', y='USD', ax=axs[0])
        axs[0].set_title(f'{label_col} by Day of Week')

        sns.violinplot(dates_df, x='month', y='USD', ax=axs[1])
        axs[1].set_title(f'{label_col} by Month')

        sns.violinplot(dates_df, x='week_of_month', y='USD', ax=axs[2])
        axs[2].set_title(f'{label_col} by Weeek of Month')

        sns.violinplot(dates_df, x='week_of_year', y='USD', ax=axs[3])
        axs[3].set_title(f'{label_col} by Weeek of Year')

        sns.violinplot(dates_df, x='year', y='USD', ax=axs[4])
        axs[4].set_title(f'{label_col} by Year')

        plt.show()


def seasonal_decompositions(df, label_col):
    ''' 
    Plots weekly, monthly, and yearly additive decompositions, if available, and trend lines for each.

    : df : pandas DataFrame containing datetime and time series information
    : label_col : time series variable of interest
    '''

    n,m = 15,8

    try:
        weekly_decomp = seasonal_decompose(df[label_col], model='additive', period=7)
    except:
        weekly_decomp = False
    try: 
        monthly_decomp = seasonal_decompose(df[label_col], model='additive', period=30)
    except:
        monthly_decomp = False
    try:
        yearly_decomp = seasonal_decompose(df[label_col], model='additive', period=365)
    except:
        yearly_decomp = False 

    with mpl.rc_context():
        mpl.rc('figure', figsize=(n,m))
        if weekly_decomp:
            weekly_decomp.plot()
            plt.title('Weekly Decomposition')
        if monthly_decomp:
            monthly_decomp.plot()
            plt.title('Monthly Decomposition')
        if yearly_decomp:
            yearly_decomp.plot()
            plt.title('Yearly Decomposition')
        plt.show()

    with mpl.rc_context():
        mpl.rc('figure', figsize=(n,m))
        plt.title(label=f'Trend Lines {label_col}')
        if weekly_decomp:
            weekly_decomp.trend.plot(label='Weekly Trend')
        if monthly_decomp:
            monthly_decomp.trend.plot(label='Monthly Trend')
        if yearly_decomp:
            yearly_decomp.trend.plot(label='Yearly Trend')
        plt.legend()
        plt.show()

'''
def periodic_kde(df, label_col, freq, dt_col:str='none', index=False):
     
    Takes pandas DataFrame with datetime column and target variable 
    and returns periodic kde plots at a specified frequency

    : df : pandas DataFrame with required columns
    : label_col : variable of interest
    : freq : 'M' for month or 'Y' for year
    : dt_col : specify column name if it's not the index
    : index : boolean, true if datetime information is stored in the index
    
    if index == True:
        df = df.reset_index().rename(columns={'index':'date'})
        dt_col = 'date'

    g = df.groupby(pd.Grouper(key=dt_col, freq=freq))
    months = [group for _, group in g]

    for idx,month in enumerate(months):
        if len(month) > 1:
            month_num = month[dt_col].dt.month.values[0]
            month_name = calendar.month_name[month_num]
            month[label_col].plot(kind='kde', label=month_name, figsize=(20,10), legend=True, title=label_col)
'''

def periodic_kde(df, label_col, freq, dt_col=None, index=False):
    if index:
        df = df.reset_index().rename(columns={'index': 'date'})
        dt_col = 'date'

    g = df.groupby(pd.Grouper(key=dt_col, freq=freq))
    for _, month in g:
        if len(month) > 1:
            month_num = month[dt_col].dt.month.iloc[0]  # Adjusted access to month
            month_name = calendar.month_name[month_num]
            month[label_col].plot(kind='kde', label=month_name, figsize=(20, 10), legend=True, title=label_col)
    plt.show()

#####################################################################################################################
            

### Prediction ###    


def create_windowed_df(df, num_periods, label_col):
    '''
    Take univariate daily time series pandas DataFrame and 
    return windowed DataFrame of size determined by num_periods

    For forecasting, the response for the resulting windowed DataFrame 
    should be the variable of the last period created

    : df : univariate pandas time series DataFrame (datetime indexed)
    : num_periods : number of times to shift the original series (number of predictors)
    : label_col : time series target variable
    '''
    df_copy = df.copy()
    df_copy = df_copy.sort_index()
    for period in range(num_periods):
        col = f'{label_col}_t{period+2}'
        df_copy[col] = df_copy[label_col].shift(-(period+1))
    df_copy = df_copy.rename(columns={label_col:f'{label_col}_t1'})
    df_copy = df_copy.drop(df_copy.tail(num_periods).index)
    return df_copy 

#####################################################################################################################