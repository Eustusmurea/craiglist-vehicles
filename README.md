# Craigslist Vehicles Time-Series Analysis

## Overview

This project involves the analysis of vehicle price data from the Craigslist Vehicles Dataset, available on Kaggle. The main objective is to perform a time-series analysis to understand temporal patterns, identify seasonal trends, and analyze demand-supply dynamics by region and vehicle type.

## Dataset

The dataset used for this analysis can be found on Kaggle: [Craigslist Vehicles Dataset](https://www.kaggle.com/datasets/mbaabuharun/craigslist-vehicles).

## Requirements

- Python 3.x
- Jupyter Notebook (or your preferred Python development environment)
- Required Python packages (install with `pip install package-name`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - statsmodels
  - pytz
  
  ## Installing opendatasets
  This tool helps get data set from an url
  
  ```python 
  !pip install opendatasets
  ```
I downloaded the dataset using opendatasets after importing

```
import opendatasets as od
download_url = 'https://www.kaggle.com/datasets/mbaabuharun/craigslist-vehicles'
od.download(download_url)
```
## craigslist-vehicles Time series project
 I started my time-series analysis project on the Craigslist Vehicles Dataset, by importing relevant packages to my Jupyter Notebook 
 
 I created a data frame from the downloaded dataset from Kaggle
 
 ```python
 # Data Manipulation and Analysis
import pandas as pd
data_filename = '.\craigslist-vehicles/craigslist_vehicles.csv'
df = pd.read_csv(data_filename, nrows=10000)  # Read the first 10,000 rows as an example
df.info()
```
# Identify numerical and categorical columns
I identified numerical and categorical columns and filled missing values. Numerical with mean and categorical with mode

```python
numerical_columns = df.select_dtypes(include=['number'])
categorical_columns = df.select_dtypes(exclude=['number'])

df[numerical_columns.columns] = df[numerical_columns.columns].fillna(df[numerical_columns.columns].median())
df[categorical_columns.columns] = df[categorical_columns.columns].fillna(df[categorical_columns.columns].mode().iloc[0])
```

I then converted the 'posting_date' column to a datetime data type and Set 'posting_date' as the Index:

```python
df['posting_date'] = pd.to_datetime(df['posting_date'])

kenya_timezone = pytz.timezone('Africa/Nairobi')
df['posting_date'] = df['posting_date'].dt.tz_convert(kenya_timezone)

df.set_index('posting_date', inplace=True)

#save clean dataframe to csv 
df.to_csv("clean_craigslist_vehicles.csv")

```
## Exploratory Data Analysis (EDA)
With clean data, I explore it using various visualizations and statistical analysis techniques. This step is crucial for understanding temporal patterns, identifying seasonal trends, and analyzing demand-supply dynamics by region and vehicle type.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Summary statistics

summary_stats = clean_data.describe()

#data distribution
plt.figure(figsize=(12, 6))
sns.histplot(clean_data['price'], kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.xlim(0, 50000)
plt.ylim(0, 1000)
plt.show()
```
## Temporal Analysis:
**Temporal Patterns:** I investigated how vehicle prices vary over time and identify any recurring patterns or trends.

**Seasonal Trends:** Seasonal analysis helps us understand whether certain months, seasons, or periods of the year have a significant impact on vehicle prices.

 **Demand-Supply Dynamics:** We analyze how the supply and demand for vehicles change over time, both by region and vehicle type.

```python
import matplotlib.dates as mdates

# Group by region and calculate the mean price

region_grouped = clean_data.groupby('region')['price'].mean().sort_values(ascending=False)
region_grouped.plot(kind='bar', figsize=(12, 6))
plt.title('Mean Price by Region')
plt.xlabel('Region')
plt.ylabel('Mean Price')

# Group by vehicle type and calculate the median price

vehicle_type_grouped = clean_data.groupby('type')['price'].median().sort_values(ascending=False)
vehicle_type_grouped.plot(kind='bar', figsize=(12, 6))
plt.title('Median Price by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Median Price')

#Correlation Analysis:

correlation_matrix = clean_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
```


## Fit an ARIMA model

**Stationarity:** We check the stationarity of the time series data, and if necessary, we perform differencing to make it stationary.

**Choosing ARIMA Orders:** We select the appropriate orders (p, d, q) for the ARIMA model by analyzing the autocorrelation (ACF) and partial autocorrelation (PACF) plots.

**Model Fitting:** We fit the ARIMA model to the data using the selected orders. We evaluate the model's performance and make necessary adjustments.

**Forecasting:** Using the fitted ARIMA model, we make forecasts for future vehicle prices.

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA model
model = ARIMA(clean_data['price'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast with the ARIMA model
forecast = model_fit.forecast(steps=10)
```

## Time-Series chart
The time-series chart provides a visual representation of how vehicle prices evolve over time. It is a crucial tool for understanding the temporal dynamics and seasonality of the dataset.

```python
plt.figure(figsize=(12, 6))
plt.plot(clean_data.index, clean_data['price'], label='Price', color='blue')
plt.title('Vehicle Prices Over Time')
plt.xlabel('Posting Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()
```
