"""
@author Melanie McCord

Just to standardize the training and test set results for the dataset.
"""

from sklearn.model_selection import train_test_split
def createTrainingTestSet(df, y_col, test_size = 0.25, random_state = 42, transform=None):
    """
    Wrapper function around sci-kit learn's training and test set function.
    Creates a training and test set, with defined hyperparameters.
    Note that it by default sets a specific test size and random state.
    Parameters:
        df: a Pandas dataframe.
        y_col: the y column.
        test_size: the size of your test set.
        random_state: the random state (for model reproducibility purposes).
        transform: a function (transforming the X column)
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[y_col], axis=1)
    y = df[y_col]
    if transform is not None:
        X = transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test