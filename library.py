import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib as mpl
import calendar
from IPython.display import display
import statsmodels.api as sm
from sklearn.decomposition import PCA
import calendar

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
            t_df = self.df.copy()
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

    :param freq: 'M' for month or 'Y' for year
        '''
    
        if self.index:
            df = self.df.reset_index().rename(columns={'index': 'date'})
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
                    plt.figure(figsize=(20, 10))
                    month[self.label_col].plot(kind='kde', label=month_name)
                    plt.title(f'{self.label_col} KDE for {month_name}')
                    plt.legend()
            plt.show()

        if freq == 'Y':
            for idx, year in enumerate(periods):
                if len(year) > 1:
                    year_num = year[dt_col].dt.year.values[0]
                    plt.figure(figsize=(20, 10))
                    year[self.label_col].plot(kind='kde', label=str(year_num))
                    plt.title(f'{self.label_col} KDE for {year_num}')
                    plt.legend()
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

# Harmonic Regression


def harmonic_regression(df, target_col, harmonic_cols):
    '''
    Fits a harmonic regression model to the specified target variable in the DataFrame.
    Assumes that sine and cosine terms for the frequencies are already calculated and present in the DataFrame.

    :param df: DataFrame containing the data
    :param target_col: Column name of the target variable
    :param harmonic_cols: List of column names of the sine and cosine terms
    :return: Fitted model and the DataFrame with predictions
    '''

    # Add a constant to the DataFrame for the intercept
    X = sm.add_constant(df[harmonic_cols])
    y = df[target_col]

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Predict using the model
    df['predicted'] = model.predict(X)

    return model, df

#Principal Component Analysis

def perform_pca(df, n_components=None):
    
    """
    Performs principal component analysis on the given dataframe.

    :param df: DataFrame containing the data
    :param n_components: Number of components to keep. If n_components is not set then all components are kept
    :return: pca object, DataFrame of the principal components
    """
    # Standardize the data
    df_standardized = (df - df.mean()) / df.std()
    
    # Initialize PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_standardized)
    
    # Create a DataFrame with the principal components
    pc_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC{i+1}' for i in range(principal_components.shape[1])],
                         index=df.index)
    
    return pca, pc_df

#PCA Plots 
def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = min(coeff.shape[0], len(labels) if labels is not None else coeff.shape[0])
    
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    plt.scatter(xs * scalex, ys * scaley, c='blue')
    
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='red', alpha=0.5)
        if labels is not None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='green', ha='center', va='center')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

#Scree Plot

def create_scree_plot(pca):
    """
    Creates a scree plot for the explained variance of the principal components.

    :param pca: PCA object
    """
    plt.figure(figsize=(14, 6))

    # Individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, color='blue')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Proportion of  Explained Variance')
    
    # Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-', color='blue')
    plt.xlabel('Principal components')
    plt.title('Cumulative Proportion of  Explained Variance')

    plt.tight_layout()
    plt.show()

#####################################################################################################################
