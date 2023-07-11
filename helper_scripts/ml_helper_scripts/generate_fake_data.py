import numpy as np


def generateFakeData():
    """
    This is a simple function to generate fake data.
    Not actually used except for testing purposes.
    Parameters: None
    Return:
        X: random binomial samples
        y: random chisquare samples
    """
    y = np.random.binomial(1, 0.2, 1000).reshape(1000, 1)
    X = np.round(np.random.chisquare(1, 1000).reshape(1000, 1))
    return X, y
