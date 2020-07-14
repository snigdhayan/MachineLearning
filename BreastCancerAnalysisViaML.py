# coding: utf-8

# Gather breast cancer data

import time

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


# Create dataframe for test statistics

data = ["ML Algorithm", "Accuracy", "#Correct", "#Incorrect", "%Unlabeled", "Training Time (ns)"]
statistics = pd.DataFrame(columns = data)


# Train logistic regression model, predict and collect statistics

from sklearn.linear_model import LogisticRegression

start_time1 = time.time_ns()
model1 = LogisticRegression(random_state=0,max_iter=10000).fit(X_train, Y_train)
training_time1 = time.time_ns() - start_time1
pred = model1.predict(X_test) == Y_test

statistics.loc[0] = ["Logistic Regression",
                     round(100*model1.score(X_test, Y_test),2), 
                     np.count_nonzero(pred==True),
                     np.count_nonzero(pred==False), 
                     0,
                     training_time1]

# Train SVM model, predict (use 'linear' as kernel) and collect statistics

from sklearn import svm

start_time2 = time.time_ns()
model2 = svm.SVC(kernel='linear').fit(X_train, Y_train)
training_time2 = time.time_ns() - start_time2
pred = model2.predict(X_test) == Y_test

statistics.loc[1] = ["Support Vector Machine",
                     round(100*model2.score(X_test, Y_test),2), 
                     np.count_nonzero(pred==True),
                     np.count_nonzero(pred==False), 
                     0,
                     training_time2]

# Train decision tree model, predict and collect statistics

from sklearn import tree

start_time3 = time.time_ns()
model3 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 5).fit(X_train,Y_train)
training_time3 = time.time_ns() - start_time3
pred = model3.predict(X_test) == Y_test

statistics.loc[2] = ["Decision Tree", 
                     round(100*model3.score(X_test, Y_test),2), 
                     np.count_nonzero(pred==True), 
                     np.count_nonzero(pred==False), 
                     0,
                     training_time3]

# Define the number of unlabeled data sets and randomly remove the labels (set them to -1)

n_unlabeled = int(X_train.shape[0] * 0.9)
idxs = np.random.choice(X_train.shape[0], replace = False, size=n_unlabeled)

y = np.asarray(Y_train)
for i in idxs:
    y[i] = -1   

Y_train = y


# Train semi-supervised Naive Bayes model from package "pomegranate", predict and collect statistics

from pomegranate import NaiveBayes, NormalDistribution

start_time4 = time.time_ns()
model4 = NaiveBayes.from_samples(NormalDistribution, X_train, Y_train, verbose=False)
training_time4 = time.time_ns() - start_time4
pred = model4.predict(X_test) == Y_test

statistics.loc[3] = ["SS Naive Bayes", 
                     round(100*model4.score(X_test, Y_test),2), 
                     np.count_nonzero(pred==True), 
                     np.count_nonzero(pred==False), 
                     round(100*idxs.size/Y_train.size,2),
                     training_time4]

# Train semi-supervised LabelSpreading model, predict (use 'knn' as kernel) and collect statistics

from sklearn.semi_supervised import LabelSpreading

start_time5 = time.time_ns()
model5 = LabelSpreading(kernel = 'knn', n_neighbors = 10, max_iter=1000).fit(X_train, Y_train)
training_time5 = time.time_ns() - start_time5
pred = model5.predict(X_test) == Y_test

statistics.loc[4] = ["SS Label Spreading", 
                     round(100*model5.score(X_test, Y_test),2), 
                     np.count_nonzero(pred==True), 
                     np.count_nonzero(pred==False), 
                     round(100*idxs.size/Y_train.size,2),
                     training_time5]

# Print summary statistics

print(statistics)


