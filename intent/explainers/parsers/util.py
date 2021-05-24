import numpy as np


def check_data(X, y=None, task='regression'):
    """
    Make sure the data is valid.
    """
    X = check_input_data(X)

    if y is not None:
        if task == 'regression':
            y = check_regression_targets(y)
        elif task in ['binary', 'multiclass']:
            y = check_classification_labels(y)
        else:
            raise ValueError(f'Unknown task {task}')
        result = (X, y)

    else:
        result = X

    return result


def check_input_data(X):
    """
    Makes sure data is of np.float32 type.
    """
    assert X.ndim == 2
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X


def check_classification_labels(y):
    """
    Makes sure labels are of np.int32 type.
    """
    assert y.ndim == 1
    if y.dtype != np.int32:
        y = y.astype(np.int32)
    return y


def check_regression_targets(y):
    """
    Makes sure regression targets are of np.float32 type.
    """
    assert y.ndim == 1
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y


def sigmoid(z):
    """
    Squashes elements in z to be between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """
    Differentiable argmax function.
    """
    if type(z) == list:
        z = np.array(z, dtype=np.float32)

    if z.ndim == 1:
        result = np.exp(z) / np.exp(z).sum()
    else:  # take softmax along axis 1
        assert z.ndim == 2
        result = np.exp(z) / (np.exp(z).sum(axis=1).reshape(-1, 1))

    return result


def logit(z):
    """
    Inverse of sigmoid.
    """
    assert isinstance(z, float)
    return np.log(z / (1 - z))
