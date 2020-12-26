# MachineLearning

I applied various machine learing techniques to analyze the breast cancer dataset (30 features and roughly 550 entries). My main interest was to ascertain the efficacy of semi-supervised learning based on the following two approaches:

1. Semi-supervised Label Spreading - https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading
2. Semi-supervised Naive Bayes - https://pomegranate.readthedocs.io/en/latest/semisupervised.html

All entries of the breast cancer dataset were initially labeled. Following the documentation I removed randomly X% of the labels from the dataset. Then I applied the algorithms with specific hyperparameters. I also compared the prediction accuracies with that of the standard supervised classification algorithms like 'logistic regression', 'support vector machine' and 'decision tree' (consult BreastCancerAnalysisViaML.py).
