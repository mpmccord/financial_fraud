
"""
@author Melanie McCord

Here is sample code that loads a model from the model directory,
runs it on a test set, and creates a confusion matrix.
"""

import matplotlib.pyplot as plt
from sklearn import metrics


def plotConfusionMatrix(y_true, y_pred):
    """
    Displays and creates a confusion matrix using scikit-learn.
    Parameters:
        y_true: the true value of the array
        y_pred: the predicted value (e.g. model output)
    Returns:
        the confusion matrix.
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    metrics.ConfusionMatrixDisplay(cm).plot()
    plt.show()

"""if __name__ == "__main__":
    clf = loadPickleFile("../../models/sample_models/sample_dtree.pickle")
    X, y_true = generateFakeData()
    y_pred = clf.predict(X)
    plotConfusionMatrix(y_true, y_pred)"""
