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

# Train SVM model, check statistics and predict (use 'rbf' as kernel)

from sklearn import svm

model = svm.SVC(kernel='rbf',probability=True).fit(X_train, Y_train)

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

