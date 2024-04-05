---
jupytext:
  formats: ipynb,py
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

TODO: 
- move these files temporarily, delete folder, re-clone, make branch, add files back and commit
- ts_lib --> OO
- explore --> returns all plots
- ml stuff (static?) - correlation analysis (heatmap, scatterplots), pca and scree, histograms
- better time series data

```{code-cell} ipython3
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
```

```{code-cell} ipython3
from ts_lib import plot_ts
from ts_lib import append_datetime_cols
from ts_lib import seasonal_catplot
from ts_lib import seasonal_decompositions
from ts_lib import periodic_kde
from ts_lib import create_windowed_df
```

```{code-cell} ipython3
# Convert the XML of the last 90 days of exchange rates from the ECB website to a pandas DataFrame

last_90 = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml?9531d63dc137832dc128c3fcc9bc4f12"

with urllib.request.urlopen(last_90) as response:
   xml_data = response.read()

root = ET.fromstring(xml_data)

parent = root.find('{http://www.ecb.int/vocabulary/2002-08-01/eurofxref}Cube')

df = pd.DataFrame(columns=['USD', 'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'CHF'],
                  dtype=np.dtype('float64'))
df.index = df.index.astype(np.dtype('datetime64[ns]'))

cube = root.find('{http://www.ecb.int/vocabulary/2002-08-01/eurofxref}Cube')

currencies_lst = ['USD', 'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'CHF']
dates = []
currencies_dict = {}

for child in cube:
    dates.append(child.attrib['time'])
    for rate_tag in child:
        currency = list(rate_tag.attrib.values())[0]
        rate = list(rate_tag.attrib.values())[1]
        if currency in currencies_lst:
            if currency in currencies_dict:
                currencies_dict[currency].append(rate)
            else:
                currencies_dict[currency] = [rate]

df = pd.DataFrame.from_dict(currencies_dict, dtype=np.dtype('float64'))
df['date'] = dates
df = df.set_index('date').rename_axis(None)
df.index = df.index.astype(np.dtype('datetime64[ns]'))

df.head()
```

```{code-cell} ipython3
# This assumes you have a DataFrame named df
df.to_csv('df_export.csv', index=False)
```

```{code-cell} ipython3
USD = df[['USD']]
```

```{code-cell} ipython3
plot_ts(USD, 'USD')
```

```{code-cell} ipython3
dates_df = append_datetime_cols(USD, index=True)
dates_df.head()
```

```{code-cell} ipython3
seasonal_catplot(USD, 'USD', kind='boxplot', index=True)
```

```{code-cell} ipython3
seasonal_catplot(USD, 'USD', kind='violinplot', index=True)
```

```{code-cell} ipython3
seasonal_decompositions(USD,'USD')
```

```{code-cell} ipython3
periodic_kde(USD, 'USD', freq='M', index=True)
```

```{code-cell} ipython3
# def make_fourier():
```

```{code-cell} ipython3
# see https://www.youtube.com/watch?v=S_Z8RnTE5dI ~52:00 minute mark

eleven_window = create_windowed_df(USD, 10, 'USD')
display(eleven_window.head(11))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = eleven_window.drop('USD_t11', axis=1)  
y = eleven_window['USD_t11']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

predictions = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)
```

```{code-cell} ipython3

```
