# coding: utf-8

# Gather breast cancer data

from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
breast_cancer_data = breast_cancer.data
breast_cancer_labels = breast_cancer.target


# Prepare data as pandas dataframe


import numpy as np
labels = np.reshape(breast_cancer_labels,(569,1))
final_breast_cancer_data = np.concatenate([breast_cancer_data,labels],axis=1)

import pandas as pd
breast_cancer_dataset = pd.DataFrame(final_breast_cancer_data)
features = breast_cancer.feature_names
features_labels = np.append(features,'label')
breast_cancer_dataset.columns = features_labels

"""
Replace 0,1 label by medical terminology (Benign = cancer false, Malignant = cancer true)

breast_cancer_dataset['label'].replace(0, 'Benign',inplace=True)
breast_cancer_dataset['label'].replace(1, 'Malignant',inplace=True)
"""

# Split data for training and testing

from sklearn.model_selection import train_test_split

split = 0.3
breast_cancer_dataset_train, breast_cancer_dataset_test = train_test_split(breast_cancer_dataset, test_size=split)
X_train, Y_train = breast_cancer_dataset_train.drop(columns='label'), breast_cancer_dataset_train['label']
X_test, Y_test = breast_cancer_dataset_test.drop(columns='label'), breast_cancer_dataset_test['label']

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

