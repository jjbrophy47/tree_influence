import numpy as np

def check_classification_data(X, y=None):
    """
    Makes sure data is of double type,
    and labels are of integer type.
    """
    result = None

    if X.dtype != np.float32:
        X = X.astype(np.float32)

    if y is not None:
        if y.dtype != np.int32:
            y = y.astype(np.int32)
        result = X, y
    else:
        result = X

    return result


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
