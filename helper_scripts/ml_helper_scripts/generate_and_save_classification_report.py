"""
@author Melanie McCord

This file provides functions to generate and save the classification report from model outputs.
"""
import pandas as pd

from sklearn.metrics import classification_report


def saveClassificationReport(y_true, y_pred, file_path):
    """
    Creates the classification report and saves it into a csv file.
    Parameters:
        y_true: the true value of y
        y_pred: the predicted value of y.
    Return:
        cr: the classification report itself as a dataframe
    """
    cr = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(file_path)
    return cr

"""
if __name__ == "__main__":
    clf = loadPickleFile("../../models/sample_models/sample_dtree.pickle")
    X, y = generateFakeData()
    y_pred = clf.predict(X)
    saveClassificationReport(y, y_pred,
                             "/home/mpmccord/Desktop/GitHub/python_projects/noc2023/models/sample_models/sample_results/fake_results.csv")
"""