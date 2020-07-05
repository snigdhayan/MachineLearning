# MachineLearningAlgorithms

I tested the efficacy of semi-supervised learning using the following two approaches:

1. Semi-supervised Label Spreading - https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading
2. Semi-supervised Naive Bayes - https://pomegranate.readthedocs.io/en/latest/semisupervised.html

I used the breast cancer analysis data (30 features and roughly 550 entries). They were initially all labeled. Following the documentation I removed randomly X% of the labels from the dataset. Then I applied the algorithms with specific hyperparameters. I also compared the prediction accuracies with that of the standard supervised classification algorithms like 'logistic regression' and 'support vector machine' (consult BreastCancerAnalysisViaML.py).
