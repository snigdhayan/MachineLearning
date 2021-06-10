#!/usr/bin/env python
# coding: utf-8


import numpy as np

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split


# Get data from OpenML for the prediction of MPG (miles per gallon)
df = fetch_openml(data_id=455, as_frame=True).data
df.dropna(axis = 0, how = 'any', inplace = True)
# df.head(10)


# Preprocess data
split = 0.2
df_train, df_test = train_test_split(df, test_size=split)

X_train, y_train = df_train.drop(columns='mpg'), df_train['mpg']
X_test, y_test = df_test.drop(columns='mpg'), df_test['mpg']


# Train model
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=7).fit(X_train, y_train)
gpr.score(X_train, y_train)


# Make predictions
predictions, std_dev = gpr.predict(X_test, return_std=True)
predictions = np.asarray(predictions)
ground_truths = np.asarray(y_test)
gpr.score(X_test, y_test)



# Visualize the results
plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')

# Plot the points
# x-axis values 
x = range(len(predictions))
# y-axis values 
y1 = predictions
plt.plot(x, y1)
plt.fill_between(x, predictions - std_dev, predictions + std_dev, alpha=0.2, color='k')

# y-axis values 
y2 = ground_truths
plt.plot(x, y2)

# x-axis label
plt.xlabel(F'Input') 

# y-axis label
plt.ylabel(F'Miles per gallon (MPG)') 

# Title
plt.title(F'Ground Truth vs. Predicted Values')

legends = [F'Ground Truths', 
           F'Predicted Values']
plt.legend(legends, loc='upper right')

plt.show()

