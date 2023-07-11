"""
@author Melanie McCord

This file provides a template for training a logistic regression model.
based on the grid search cv hyperparameter tuning.

Note that for this particular model, in addition to the base features,
multiple transformations are performed on the dataset.

In the example below, the model is trained on a set of parameters, then saved to a pickle file.

"""
from ml_helper_scripts.generate_and_save_classification_report import saveClassificationReport
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from ml_helper_scripts.create_training_test_set import createTrainingTestSet
from ml_helper_scripts.saving_and_loading_pickle_files import savePickleFile
from sklearn.model_selection import GridSearchCV
import pandas as pd


def buildMultinomialLogisticRegressionClassifier(X_train, y_train):
    """
    Builds a decision tree classifier and identifies the best set of parameters.
    Parameters:
        X_train: the training X values.
        y_train: the training y values.
    Returns:
        clf: a decision tree classifier, optimized by gridsearchcv.
    """
    clf = LogisticRegression()
    param_grid = {'penalty': ['l1', 'l2', None],
                  'solver': ["newton-cg", "sag", "saga", "lbfgs"],
                  'fit_intercept': [False, True]
                  }
    grid = GridSearchCV(clf, param_grid=param_grid)
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    print(best_clf.get_params())
    return best_clf


if __name__ == "__main__":
    financial_data = pd.read_csv("/home/mpmccord/Desktop/GitHub/python_projects/financial_fraud/Data/PS_20174392719_1491204439457_log.csv")
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = createTrainingTestSet(financial_data, "isFraud")
    best_clf = buildMultinomialLogisticRegressionClassifier(X_train, y_train)
    savePickleFile(best_clf, "/home/mpmccord/Desktop/GitHub/python_projects/noc2023/models/1p_2p_3p_4p_5p/lr.pickle")

    X_train, X_test, y_train, y_test = createTrainingTestSet(financial_data, "isFraud", transform=lambda x: np.log(x + 1))
    best_clf = buildMultinomialLogisticRegressionClassifier(X_train, y_train)
    savePickleFile(best_clf, "/home/mpmccord/Desktop/GitHub/python_projects/noc2023/models/1p_2p_3p_4p_5p/lr_log.pickle")

    X_train, X_test, y_train, y_test = createTrainingTestSet(financial_data, "isFraud", transform=sc.fit_transform)
    best_clf = buildMultinomialLogisticRegressionClassifier(X_train, y_train)
    savePickleFile(best_clf, "/home/mpmccord/Desktop/GitHub/python_projects/noc2023/models/1p_2p_3p_4p_5p/lr_scaled.pickle")

    # Evaluating and Saving Precision, Recall, And Accuracy
    y_pred = best_clf.predict(X_test)
    saveClassificationReport(y_test, y_pred, "/home/mpmccord/Desktop/GitHub/python_projects/financial_fraud/classification_reports/lr_test.csv")
    y_pred = best_clf.predict(X_train)
    saveClassificationReport(y_train, y_pred, "/home/mpmccord/Desktop/GitHub/python_projects/financial_fraud/classification_reports/lr_train.csv")


#%%
