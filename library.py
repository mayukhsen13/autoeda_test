import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib as mpl
import calendar
from IPython.display import display


#####################################################################################################################


### Automated Exploratory Data Analysis ###

class AutoEDA:
    
    def __init__(self, df:pd.DataFrame, label_col:str='none', dt_col:str='none', dt_format:str='%Y-%m-%d', index=True, return_dt_index=True):

        ''' 
        : df : pd.DataFrame with 'datetime64[ns]' either as the index or in one of its columns
        : label_col : variable of interest
        : dt_col : specified datetime indexed column with dt_format string format
        : dt_format : string format for col
        : index : boolean, true if Dataframe is datetime indexec (and no datetime column)
        : return_dt_index : boolean, if original datetime column (or index) is wished to be returned as the DataFrame index
        '''

        self.df = df
        self.label_col = label_col
        self.dt_col = dt_col
        self.dt_format = dt_format
        self.index = index 
        self.return_dt_index = return_dt_index

    def plot_ts(self): 

        '''
        Returns time series lineplot from univariate time series variable 
        '''

        self.df[self.label_col].plot(figsize=(20,10), title=self.label_col)
        plt.show()

    def append_datetime_cols(self) -> pd.DataFrame:

        '''
        Extracts datetime components from daily datetime column of a pandas DataFrame.
        If DataFrame has no datetime column but is datetime indexed, index=True
        Returns original pandas DataFrame augmented with datetime components.
        '''
        
        if self.index == True:
            t_df = self.df.reset_index().rename(columns={'index':'date'})
            dt_col_name = 'date'
        else:
            t_df = df.copy()
            dt_col_name = self.dt_col

        dt_col = pd.to_datetime(t_df[dt_col_name],
                                format=self.dt_format,
                                errors='coerce')
        t_df[dt_col_name] = dt_col
        t_df['year'], t_df['month'], t_df['day'] = dt_col.dt.year, dt_col.dt.month, dt_col.dt.day
        t_df['week_of_year'] = dt_col.dt.isocalendar().week
        t_df['week_of_month'] = dt_col.apply(lambda d: (d.day-1) // 7 + 1)
        t_df['day_of_week'] = dt_col.dt.dayofweek.apply(lambda d: d+1)

        if self.return_dt_index == True:
            t_df = t_df.set_index(dt_col_name)
            t_df.index.name = None

        return t_df
    
    def seasonal_catplots(self, kind):

        ''' 
        Takes univariate daily time series pandas DataFrame and returns seasonal datetime categorical plots

        : kind : either 'boxplot' or 'violinplot' 
        '''

        dates_df = self.append_datetime_cols()

        if kind=='boxplot':
            fig, axs = plt.subplots(5, 1, figsize=(20,30))
            sns.boxplot(dates_df, x='day_of_week', y=self.label_col, ax=axs[0])
            axs[0].set_title(f'{self.label_col} by Day of Week')

            sns.boxplot(dates_df, x='month', y=self.label_col, ax=axs[1])
            axs[1].set_title(f'{self.label_col} by Month')

            sns.boxplot(dates_df, x='week_of_month', y=self.label_col, ax=axs[2])
            axs[2].set_title(f'{self.label_col} by Week of Month')

            sns.boxplot(dates_df, x='week_of_year', y=self.label_col, ax=axs[3])
            axs[3].set_title(f'{self.label_col} by Week of Year')

            sns.boxplot(dates_df, x='year', y=self.label_col, ax=axs[4])
            axs[4].set_title(f'{self.label_col} by Year')

            plt.show()

        if kind=='violinplot':
            fig, axs = plt.subplots(5, 1, figsize=(20,30))
            sns.violinplot(dates_df, x='day_of_week', y=self.label_col, ax=axs[0])
            axs[0].set_title(f'{self.label_col} by Day of Week')

            sns.violinplot(dates_df, x='month', y=self.label_col, ax=axs[1])
            axs[1].set_title(f'{self.label_col} by Month')

            sns.violinplot(dates_df, x='week_of_month', y=self.label_col, ax=axs[2])
            axs[2].set_title(f'{self.label_col} by Week of Month')

            sns.violinplot(dates_df, x='week_of_year', y=self.label_col, ax=axs[3])
            axs[3].set_title(f'{self.label_col} by Week of Year')

            sns.violinplot(dates_df, x='year', y=self.label_col, ax=axs[4])
            axs[4].set_title(f'{self.label_col} by Year')

            plt.show()

    def seasonal_decompositions(self):

        ''' 
        Plots weekly, yearly, and yearly additive decompositions, if available, and trend lines for each.
        '''
        
        n,m = 15,8

        try:
            weekly_decomp = seasonal_decompose(self.df[self.label_col], model='additive', period=7)
        except:
            weekly_decomp = False
        try: 
            monthly_decomp = seasonal_decompose(self.df[self.label_col], model='additive', period=30)
        except:
            monthly_decomp = False
        try:
            yearly_decomp = seasonal_decompose(self.df[self.label_col], model='additive', period=365)
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
            plt.title(label=f'Trend Lines {self.label_col}')
            if weekly_decomp:
                weekly_decomp.trend.plot(label='Weekly Trend')
            if monthly_decomp:
                monthly_decomp.trend.plot(label='Monthly Trend')
            if yearly_decomp:
                yearly_decomp.trend.plot(label='Yearly Trend')
            #plt.legend()
            plt.show()

    def periodic_kde(self, freq):

        ''' 
        Takes pandas DataFrame with datetime column and target variable 
        and returns periodic kde plots at a specified frequency

        : freq : 'M' for month or 'Y' for year
        '''

        if self.index == True:
            df = self.df.reset_index().rename(columns={'index':'date'})
            dt_col = 'date'
            
        else:
            df = self.df
            dt_col = self.dt_col

        g = df.groupby(pd.Grouper(key=dt_col, freq=freq))
        periods = [group for _, group in g]

        if freq == 'M':
            for idx, month in enumerate(periods):
                if len(month) > 1:
                    month_num = month[dt_col].dt.month.values[0]
                    month_name = calendar.month_name[month_num]
                    month[self.label_col].plot(kind='kde', label=month_name, figsize=(20,10), legend=True, title=self.label_col)
            plt.show()

        if freq == 'Y':
            for idx,year in enumerate(periods):
                if len(year) > 1:
                    year_num = year[dt_col].dt.year.values[0]
                    year[self.label_col].plot(kind='kde', label=year_num, figsize=(20,10), legend=True, title=self.label_col)
            plt.show()

def Explore(df, label_col):
    display(df.describe().reset_index()) # display for notebooks
    obj = AutoEDA(df, label_col)
    obj.plot_ts()
    obj.seasonal_catplots(kind='boxplot')
    obj.seasonal_catplots(kind='violinplot')
    obj.seasonal_decompositions()
    obj.periodic_kde(freq='M')
    obj.periodic_kde(freq='Y')


#####################################################################################################################
            

### Prediction ###    


def create_windowed_df(df, num_periods, label_col):
    '''
    Take univariate daily time series pandas DataFrame and 
    return windowed DataFrame of size determined by num_periods

    : df : univariate pandas time series DataFrame (datetime indexed)
    : num_periods : number of times to shift the original series (number of predictors)
    : label_col : time series target variable
    '''
    df_copy = df.copy()
    df_copy = df_copy.sort_index()
    for period in range(num_periods):
        col = f'{label_col}_t{-(period+1)}'
        df_copy[col] = df_copy[label_col].shift(period+1)
    df_copy = df_copy.drop(df_copy.head(num_periods).index)
    df_copy = df_copy[df_copy.columns[::-1]]
    return df_copy 

def make_fourier(df:pd.DataFrame, periodic_cols:list, K:int=1) -> pd.DataFrame:
    ''' 
    see: https://otexts.com/fpp3/useful-predictors.html#fourier-series and https://otexts.com/fpp2/dhr.html 

    : df : pandas DataFrame containing predictors (some periodic) and a response (ideally continuous)
    : periodic_cols : a list containing the names of the periodic columns of the DataFrame
    : K : smoothing parameter, determines the number of sine, cosine pairs for each periodic column
    '''
    for col in periodic_cols:
        m = df[col].max()
        for k in range(1, K+1):
            df[f'{col}_sin{k}'] = np.sin(2*np.pi*k*df[col] / m)
            df[f'{col}_cos{k}'] = np.cos(2*np.pi*k*df[col] / m)
        df = df.drop(columns=[col])
    return df


#####################################################################################################################
