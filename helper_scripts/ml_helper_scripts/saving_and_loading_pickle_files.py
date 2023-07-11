"""
@author Melanie McCord

Here are the two functions to save and load a pickle file, along with a sample unit test.
"""
import pickle


def savePickleFile(mdl, file_path):
    """
    Writes a Python object to a pickle file.
    Parameters:
        mdl: the Python object you want to save.
        file_path: the file path where you want to save your object.
    Returns:
        mdl: the original object.
    """
    with open(file_path, "wb") as handle:
        pickle.dump(mdl, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return mdl


def loadPickleFile(file_path):
    """
    Loads a pickle file given a data path.
    Parameters:
        file_path: file path where your data is stored.

    Returns:
        the extracted pickle object saved at the file path.
    """
    with open(file_path, "rb") as handle:
        mdl = pickle.load(handle)
    return mdl
