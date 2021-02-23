# MachineLearning

I applied various machine learing techniques to analyze the breast cancer dataset (`30 features` and `569 entries`). My main interest was to ascertain the efficacy of `semi-supervised learning` based on the following two approaches:

1. Semi-supervised `Label Spreading` - https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading
2. Semi-supervised `Naive Bayes` - https://pomegranate.readthedocs.io/en/latest/semisupervised.html

All entries of the breast cancer dataset were initially labeled. Following the documentation I removed randomly `X%` of the labels from the dataset, so that the true benefit of semi-supervised learning is discernible. I also compared the prediction accuracies with those of the standard supervised classification algorithms like `logistic regression`, `support vector machine` and `decision tree` (consult `BreastCancerAnalysisViaML.py`).

In `CreditCardFraudDetection.py` I used the `Local Outlier Factor (LOF)` model (consult https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor) to predict credit card fraud based on the kaggle dataset - https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv. Using this approach I was able to achieve an accuracy above 90%.
