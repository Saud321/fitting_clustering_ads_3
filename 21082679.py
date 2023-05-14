# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from statsmodels.tsa.stattools import adfuller

"""The adfuller function from the statsmodels.tsa.stattools library is imported by this code.
A standard statistical test for a time series' stationarity, the Augmented Dickey-Fuller unit 
root test, is carried out using this function."""

# seasonal_decompose() is used for time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Import custom module
import cluster_tools as ct

# Setting the specific columns
columnsName = ["DATE", "value"]

# Set the option to display all columns
pd.set_option('display.max_columns', None)

# Read in the "Electric_Production.csv" file as a pandas dataframe and setting the specific columns
electric_production_df = pd.read_csv("Electric_Production.csv", names=columnsName, header=0, parse_dates=[0])

# storing in the array
array = electric_production_df.to_numpy()

# Transpose of dataset
transposed_data = array.T

# print Transpose
print(transposed_data)

# Display the first 5 rows of the DataFrame
electric_production_df.head()

# This code displays the last five rows of a DataFrame
electric_production_df.tail()

# This code generates descriptive statistics of a Dataframe
electric_production_df.describe()

# filling the null values if there is some null values
electric_production_df = electric_production_df.fillna(0)

# Checking the info for further details to check the null or non-null values
electric_production_df.info()

# Again verifying the dataset to check the dataset
electric_production_df.head()

# Setting the date format to Year-Month-Day
electric_production_df['DATE'] = pd.to_datetime(electric_production_df['DATE'], infer_datetime_format=True)

# Setting the date as an index
electric_production_df = electric_production_df.set_index(['DATE'])

# Verifying the dataset
electric_production_df.head()

# Calculate the rolling mean and rolling std according to months
rolling_mean = electric_production_df.rolling(window=12).mean()
rolling_std = electric_production_df.rolling(window=12).std()

# Plotting the rolling mean and std
# Setting the figuresize
plt.figure(figsize=(12, 8), dpi=300)

# displaying the plot
plt.plot(electric_production_df, label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')

# Setting the x label and y label
plt.xlabel('Date', size=12)
plt.ylabel('Electric Production', size=12)

# Setting the legend at the upper left position
plt.legend(loc='upper left')

# setting the super title and title
plt.suptitle('Rolling Statistics', size=14)
plt.title("21082679")

# Saving the plot
plt.savefig("RollingOutput.png")

# displaying the plot
plt.show()

# Use the augmented Dickey-Fuller test to check for stationarity
adful = adfuller(electric_production_df, autolag="AIC")

# Create a DataFrame with ADF test results
output_df = pd.DataFrame({
    "Values": [adful[0], adful[1], adful[2], adful[3], adful[4]['1%'], adful[4]['5%'], adful[4]['10%']],
    "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",
               "critical value (1%)", "critical value (5%)", "critical value (10%)"]
})

# Print the DataFrame
print(output_df)

# Calculate autocorrelation at lag 1
autocor_lag1 = electric_production_df['value'].autocorr(lag=1)

# Print the result
print("One Month Lag: ", autocor_lag1)

# Calculate autocorrelation for lags of 3, 6, and 9 months
autocorr_lag3 = electric_production_df['value'].autocorr(lag=3)
autocorr_lag6 = electric_production_df['value'].autocorr(lag=6)
autocorr_lag9 = electric_production_df['value'].autocorr(lag=9)

# Print the results
print("Three Month Lag:", autocorr_lag3)
print("Nine Month Lag:", autocorr_lag9)

# Perform seasonal decomposition of time series data to calculate the useful insights
decompose = seasonal_decompose(electric_production_df['value'], model='additive', period=7)

# Plot decomposition
decompose.plot()

# Saving and showing the figure
plt.savefig("21082679-Saud.png", dpi=300)
plt.show()

# Read the dataset from a CSV file
data = pd.read_csv("agridataset.csv")

"""On a Pandas DataFrame with the identifier "data," this code is invoking the "head()" function.
The 'head()' function, where n is by default 5, retrieves the DataFrame's top n rows.
The console is then output with the resulting DataFrame."""

# filling the null values which can disturb the outcomes
data = data.fillna(0)

# Selecting columns from dataframe
sec_data = data[['1970', '1980', '2010', '2020']]

corr = sec_data.corr()

print(corr)

# Display the scatter matrix plot
plt.show()

# Selecting '1970' and '2020' columns from 'sec_data' DataFrame
df_ex = sec_data[['1970', '2020']]

# Dropping rows with null values
df_ex = df_ex.dropna()

# Resetting index
df_ex = df_ex.reset_index()

# Printing first 15 rows of the DataFrame
print(df_ex.iloc[0:15])

# Dropping 'index' column
df_ex = df_ex.drop('index', axis=1)

# Printing first 15 rows of the DataFrame
print(df_ex.iloc[0:15])

# Scale the dataframe
df_norm, df_min, df_max = ct.scaler(df_ex)

for ncluster in range(2, 10):
    # set up the  cluster with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # fitting the dataset
    kmeans.fit(df_norm)
    labels = kmeans.labels_

    cen = kmeans.cluster_centers_

    print(ncluster, skmet.silhouette_score(df_ex, labels))

# Set number of clusters
ncluster = 3

# Perform KMeans clustering
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# # Extract x and y coordinates of cluster centers
cen = np.array(cen)

# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print(scen)

xcen = scen[:, 0]
ycen = scen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_ex["1970"], df_ex["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")

plt.suptitle("Three Centered Clusters", size=20)
plt.title("21082679", size=18)
plt.xlabel("Agriculture(1970)", size=16)
plt.ylabel("Agriculture(2020)", size=16)
plt.savefig("Three Centered Clusters.png", dpi=300)
plt.show()
