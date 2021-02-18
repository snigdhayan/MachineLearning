#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import time

# Read and downsize dataset to reduce computation

credit_card_dataset = pd.read_csv("./credit_card_dataset.csv")
credit_card_dataset = credit_card_dataset.sample(50000, random_state=123)
credit_card_dataset = credit_card_dataset.drop(columns="Time")


# Adjust labels to fit the prediction output of Local Outlier Factor (LOF) model

credit_card_dataset["Class"] = credit_card_dataset["Class"].apply(lambda x:1 if x==0 else -1)
dataset, ground_truth = credit_card_dataset.drop(columns="Class"), credit_card_dataset["Class"]

# Isolate the outliers in the dataset

outliers_dataset = credit_card_dataset[ground_truth==-1]


import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Fit LOF model for outlier detection only

model = LocalOutlierFactor(n_neighbors=100, contamination=0.1, novelty=False)


# Compute the predicted labels of the samples and the associated time training time

start_time = time.time_ns()
class_pred = model.fit_predict(dataset)
training_time = time.time_ns() - start_time
print("\nTraining time: {} seconds\n".format(round(training_time/1000000000,2)))


# Compute the accuracy of prediction

n_errors = (class_pred != ground_truth).sum()
X_scores = model.negative_outlier_factor_
print("\nPrediction accuracy: {}%\n".format(round(100*(len(class_pred)-n_errors)/len(class_pred),2)))


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2


# Normalize dataset to with MinMaxScaler

dataset_norm = MinMaxScaler().fit_transform(dataset)


# Identify the top two most relevant features

selector = SelectKBest(chi2, k=2)
dataset_new = selector.fit_transform(dataset_norm, ground_truth)

feature_support = selector.get_support()
selected_features = dataset.loc[:,feature_support].columns


# Prepare dataset and outliers therein with reduced features as dataframes

df = pd.DataFrame(data = dataset, columns = selected_features)
outliers_df = pd.DataFrame(data = outliers_dataset, columns = selected_features)

# Select the top two features for visualization

X1 = df[selected_features[0]]
X2 = df[selected_features[1]]

Y1 = outliers_df[selected_features[0]]
Y2 = outliers_df[selected_features[1]]


# Plot all data points according to the top two selected features

plt.title("Local Outlier Factor (LOF)")
plt.scatter(X1, X2, 
            color="g", 
            s=3, 
            label="Data points")

# Plot red circles around outliers in the dataset according to ground truth

plt.scatter(Y1, Y2, 
            s=30,
            edgecolors="r",
            facecolors="none", 
            label="Outliers")
plt.axis("tight")
plt.xlim((1.1*min(X1), 1.1*max(X1)))
plt.ylim((1.1*min(X2), 1.1*max(X2)))
plt.xlabel("No. of prediction errors: %d" % (n_errors))
legend = plt.legend(loc="upper left")
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()