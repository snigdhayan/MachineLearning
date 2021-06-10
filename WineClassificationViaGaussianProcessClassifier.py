#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split


# Get data from OpenML for the prediction of wine label
df = fetch_openml(data_id=187, as_frame=True).data
labels = fetch_openml(data_id=187, as_frame=True).target
df['Label'] = labels
df.dropna(axis = 0, how = 'any', inplace = True)


# Preprocess data
split = 0.5
df_train, df_test = train_test_split(df, test_size=split)

X_train, Y_train = df_train.drop(columns='Label'), df_train['Label']
X_test, Y_test = df_test.drop(columns='Label'), df_test['Label']

# Train model
kernel = 1.0 * RBF([1.0])
model = GaussianProcessClassifier(kernel=kernel, random_state=7).fit(X_train, Y_train)
print("Percentage of correct predictions while training = {}".format(round(100*model.score(X_test, Y_test),2)))

# Make predictions
pred = model.predict(X_test) == Y_test
print("Correct: {}".format(np.count_nonzero(pred==True)),"/",
      "Incorrect: {}".format(np.count_nonzero(pred==False)))

Z1 = model.predict(X_test).reshape(Y_test.size,1)
Z2 = np.asarray(Y_test).reshape(Y_test.size,1)
Z3 = np.around(model.predict_proba(X_test),decimals=2)
data = np.concatenate((Z1,Z2,Z3),axis=1)
outcome = pd.DataFrame(data, columns = ["Predicted Label", 
                                        "Actual Label", 
                                        "Prob. Label = 1", 
                                        "Prob. Label = 2",
                                        "Prob. Label = 3"])
indicesToKeep = outcome["Predicted Label"] != outcome["Actual Label"]

print("False predictions with associated class probabilities:\n{}".format(outcome[indicesToKeep]))

# Visualize the results
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel(df.columns[0],fontsize=20)
plt.ylabel(df.columns[1],fontsize=20)
plt.title(F'Plot according to {df.columns[0]} and {df.columns[1]}',fontsize=20)
targets = [False, True]
legends = ['Incorrect prediction', 'Correct prediction']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = pred[pred == target].index
    plt.scatter(df.loc[indicesToKeep, df.columns[0]],
                df.loc[indicesToKeep, df.columns[1]], c = color, s = 50)

plt.legend(legends, prop={'size': 15}, loc='upper right')
plt.show()