# coding: utf-8

# Gather breast cancer data


from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
breast_data = breast.data
breast_labels = breast.target


# Prepare data as pandas dataframe


import numpy as np
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)

import pandas as pd
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features_labels = np.append(features,'label')
breast_dataset.columns = features_labels


# Split data for training and testing

from sklearn.model_selection import train_test_split

split = 0.3
breast_dataset_train, breast_dataset_test = train_test_split(breast_dataset, test_size=split)
X_train, Y_train = breast_dataset_train.drop(columns='label'), breast_dataset_train['label']
X_test, Y_test = breast_dataset_test.drop(columns='label'), breast_dataset_test['label']

# Randomly remove some labels (set the labels to -1)

n_unlabeled = int(X_train.shape[0] * 0.9)
idxs = np.random.choice(X_train.shape[0], replace = False, size=n_unlabeled)

y = np.asarray(Y_train)
for i in idxs:
    y[i] = -1

Y_train = y


# Train model and print statistics (use 'knn' as kernel)

from sklearn.semi_supervised import LabelSpreading

model = LabelSpreading(kernel = 'knn', n_neighbors = 10, max_iter=1000).fit(X_train, Y_train)

print("Percentage of correct predictions = {}".format(round(100*model.score(X_test, Y_test),2)))
pred = model.predict(X_test) == Y_test
print("Correct: {}".format(np.count_nonzero(pred==True)),"/",
      "Incorrect: {}".format(np.count_nonzero(pred==False)))

Z1 = model.predict(X_test).reshape(Y_test.size,1)
Z2 = np.asarray(Y_test).reshape(Y_test.size,1)
Z3 = np.around(model.predict_proba(X_test),decimals=2)
data = np.concatenate((Z1,Z2,Z3),axis=1)
outcome = pd.DataFrame(data, columns = ["Predicted Label", 
                                        "Actual Label", 
                                        "Prob. Label = 0.0", 
                                        "Prob. Label = 1.0"])
indicesToKeep = outcome["Predicted Label"] != outcome["Actual Label"]

print("False predictions with associated class probabilities:\n{}".format(outcome[indicesToKeep]))

# Plot predictions

import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel(breast_dataset.columns[0],fontsize=20)
plt.ylabel(breast_dataset.columns[1],fontsize=20)
plt.title("Plot according to mean radius and mean area",fontsize=20)
targets = [False, True]
legends = ['Incorrect prediction', 'Correct prediction']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = pred[pred == target].index
    plt.scatter(breast_dataset.loc[indicesToKeep, breast_dataset.columns[0]]
               , breast_dataset.loc[indicesToKeep, breast_dataset.columns[1]], c = color, s = 50)

plt.legend(legends,prop={'size': 15})








