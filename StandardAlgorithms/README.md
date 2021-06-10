# MachineLearning

I analyzed the breast cancer dataset (`30 features` and `569 entries`) using some standard machine learning algorithms.

1. Semi-supervised `Label Spreading` - https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading
2. Semi-supervised `Naive Bayes` - https://pomegranate.readthedocs.io/en/latest/semisupervised.html

All entries of the breast cancer dataset were initially labeled. Following the documentation I removed randomly `X%` of the labels from the dataset, so that the true benefit of semi-supervised learning is discernible. 
1. `Logistic regression`, 
2. `Support vector machine` and 
3. `Decision tree`

I also compared their prediction accuracies with those of some semi-supervised learning approaches (consult `BreastCancerAnalysisViaML.py`).