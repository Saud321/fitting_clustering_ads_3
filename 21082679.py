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

columns_name = ["DATE", "value"]

# Set the option to display all columns

pd.set_option('display.max_columns', None)


# Read in the "Electric_Production.csv" file as a pandas dataframe and setting the specific columns

df = pd.read_csv("Electric_Production.csv", names=columns_name, header=0, parse_dates=[0])


# storing in the array

array = df.to_numpy()

# Transpose of dataset

transposed_data = array.T

# print Transpose

print(transposed_data)

# Display the first 5 rows of the DataFrame

df.head()

# This code displays the last five rows of a DataFrame

df.tail()

# This code generates descriptive statistics of a Dataframe

df.describe()

# filling the null values if there is some null values

df = df.fillna(0)

# Checking the info for further details to check the null or non-null values

df.info()

# Again verifying the dataset to check the dataset

df.head()

# Setting the date format to Year-Month-Day

df['DATE'] = pd.to_datetime(df['DATE'], infer_datetime_format=True)

# Setting the date as an index

df = df.set_index(['DATE'])

# Verifying the dataset

df.head()

# Calculate the rolling mean and rolling std according to months

rolling_mean = df.rolling(window=12).mean()
rolling_std = df.rolling(window=12).std()

# Setting the figuresize

plt.figure(figsize=(12, 8), dpi=300)

# displaying the plot

plt.plot(df, label='Original')
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

# Setting the figuresize

plt.figure(figsize=(12, 8))

# displaying the plot

plt.plot(df['value'])

# Setting the x label and y label

plt.xlabel("Dates")
plt.ylabel("Electrical Production")

# setting the super title and title

plt.suptitle("Electrical Production TimeSeries")
plt.title("21082679")

# displaying the plot

plt.show()

# Use the augmented Dickey-Fuller test to check for stationarity

ad_ful = adfuller(df, autolag="AIC")

# Create a DataFrame with ADF test results

output_df = pd.DataFrame({
    "Values": [ad_ful[0], ad_ful[1], ad_ful[2], ad_ful[3], ad_ful[4]['1%'], ad_ful[4]['5%'], ad_ful[4]['10%']],
    "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",               
               "critical value (1%)", "critical value (5%)", "critical value (10%)"]
})

# Print the DataFrame

print(output_df)

# Calculate auto correlation at lag 1

auto_cor_lag1 = df['value'].autocorr(lag=1)

# Print the result

print("One Month Lag: ", auto_cor_lag1)


# Calculate auto correlation for lags of 3, 6, and 9 months

autocorr_lag3 = df['value'].autocorr(lag=3)
autocorr_lag6 = df['value'].autocorr(lag=6)
autocorr_lag9 = df['value'].autocorr(lag=9)

# Print the results

print("Three Month Lag:", autocorr_lag3)
print("Nine Month Lag:", autocorr_lag9)

# Perform seasonal decomposition of time series data to calculate the useful insights

decompose = seasonal_decompose(df['value'], model='additive', period=7)

# Plot decomposition

decompose.plot()

# Saving and showing the figure

plt.savefig("21082679-Saud.png", dpi=300)

plt.show()

# Read the dataset from a CSV file
data = pd.read_csv("agridataset.csv")


"""On a Pandas DataFrame with the identifier "data," this code is invoking the "head()" function.
The 'head()' function, where n is by default 5, retrieves the DataFrame's top n rows.
The console is then output with the resulting DataFrame.
."""

data.head()

# filling the null values which can disturb the outcomes

data = data.fillna(0)

# Selecting columns from dataframe

sec_data = data[['1970', '1980', '2010', '2020']]


corr = sec_data.corr()

print(corr)

ct.map_corr(sec_data)
# Saving the plot and showing plot

plt.savefig("heatmap.png")

plt.show()

# Plot a scatter matrix of sec_data

pd.plotting.scatter_matrix(sec_data, figsize=(12, 12), s=5, alpha=0.8)

# Save the scatter matrix plot as an image file

plt.savefig("Matrix.png", dpi=300)

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

print('n  value')

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

# Extract x and y coordinates of cluster centers
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]

# Create scatter plot with labeled points and cluster centers
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm['1970'], df_norm['2020'], 10, labels, marker='o', cmap=cm)
plt.scatter(xcen, ycen, 45, 'k', marker='d')
plt.suptitle("Three Clusters", size=20)
plt.title("21082679", size=18)
plt.xlabel("Agriculture(1970)", size=16)
plt.ylabel("Agriculture(2020)", size=16)
plt.savefig("Three Clusters.png", dpi=300)
plt.show()

print(cen)

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
