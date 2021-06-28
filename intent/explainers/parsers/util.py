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
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """
    Differentiable argmax function.

    Input
        z: 2d array of values.

    Returns 2d array of probability distributions of shape=z.shape.
    """
    if type(z) == list:
        z = np.array(z, dtype=np.float32)

    if z.ndim == 1:
        z = z.reshape(1, -1)  # shape=(1, len(z))

    centered_exponent = np.exp(z - np.max(z, axis=1, keepdims=True))
    return centered_exponent / np.sum(centered_exponent, axis=1, keepdims=True)


def logsumexp(z):
    """
    Input
        z: 2d array of values.

    Returns 2d array of normalization constants in log space, shape=(1, len(z)).
    """
    if z.ndim == 1:
        z = z.reshape(1, -1)  # shape=(1, len(z))

    maximum = np.max(z, axis=1, keepdims=True)
    return maximum + np.log(np.sum(np.exp(z - maximum), axis=1, keepdims=True))


def logit(z):
    """
    Inverse of sigmoid.
    """
    return np.log(z / (1 - z))


def to_np(x):
    """
    Convert torch tensor to numpy array.
    """
    return x.data.cpu().numpy()


class SquaredLoss(object):
    """
    Squared loss.

    Modified from:
        - https://github.com/bsharchilev/influence_boosting/blob/master/influence_boosting/loss.py
    """

    def __call__(self, y, y_raw):
        """
        Input
            y: 1d array of regression values.
            y_raw: 1d array of predicted values.

        Returns 1d array of mean-squared error losses.

        Note:
            - y and yhat could be swapped and the gradient would still be the same.
            - y_raw and y_hat are equivalent.
        """
        return 0.5 * (y - y_raw) ** 2

    def gradient(self, y, y_raw):
        """
        Input
            y: 1d array of regression values.
            y_raw: 1d array of predicted values.

        Returns 1d array of gradients w.r.t. the prediction.
        """
        return y_raw - y

    def hessian(self, y, yhat):
        """
        Input
            y: 1d array of regression values.
            y_raw: 1d array of predicted values.

        Returns 1d array of second-order derivatives w.r.t. the prediction.
        """
        return np.ones_like(y)

    def third(self, y, yhat):
        """
        Input
            y: 1d array of regression values.
            y_raw: 1d array of predicted values.

        Returns 1d array of third-order derivatives w.r.t. the prediction.
        """
        return np.zeros_like(y)


class LogisticLoss(object):
    """
    Sigmoid + Binary Cross-entropy.

    A.K.A. log loss, binomial deviance, binary objective.

    Inputs are unnormalized log probs.

    Modified from:
        - https://github.com/bsharchilev/influence_boosting/blob/master/influence_boosting/loss.py
        - https://github.com/eriklindernoren/ML-From-Scratch/blob/
            a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/supervised_learning/xgboost.py
    """

    def __call__(self, y, y_raw, eps=1e-15):
        """
        Clip yhat to just above 0 and just below 1
        since log is undefined for 0.

        Input
            y: 1d array of 0 and 1 labels.
            y_raw: 1d array of unnormalized log probs.

        Return 1d array of neg. log losses.
        """
        assert np.all(np.unique(y) == np.array([0, 1]))
        assert y.ndim == 1
        assert yhat.ndim == 1

        y_hat = sigmoid(y_raw)
        y_hat = np.clip(y_hat, eps, 1 - eps)  # prevent log(0)
        losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        return losses

    def gradient(self, y, y_raw):
        """
        Input
            y: 1d array of 0 and 1 labels.
            yhat: 1d array of pre-activation values.

        Returns 1d array of gradients w.r.t. the prediction.
        """
        y_hat = sigmoid(y_raw)
        return y_hat - y

    def hessian(self, y, y_raw):
        """
        Input
            y: 1d array of 0 and 1 labels.
            yhat: 1d array of pre-activation values.

        Returns 1d array of second-order gradients w.r.t. the prediction.
        """
        y_hat = sigmoid(y_raw)
        return y_hat * (1 - y_hat)

    def third(self, y, y_raw):
        """
        Input
            y: 1d array of 0 and 1 labels.
            yhat: 1d array of pre-activation values.

        Returns 1d array of third-order gradients w.r.t. the prediction.
        """
        y_hat = sigmoid(y_raw)
        return y_hat * (1 - y_hat) * (1 - 2 * y_hat)


class SoftmaxLoss(object):
    """
    Softmax + Cross-entropy.

    A.K.A. Multiclass log loss, multinomial deviance, multiclass objective.

    Inputs are unnormalized log probs.

    Modified from:
        - https://github.com/bsharchilev/influence_boosting/blob/master/influence_boosting/loss.py
    """

    def __call__(self, y, y_raw):
        """
        Input
            y: 2d array of one-hot-encoded labels; shape=(no. examples, no. classes).
            y_raw: 2d array of unnormalized log probs.; shape=(no. examples, no. classes).

        Return losses of shape=y_raw.shape.
        """
        y_raw_norm = y_raw - logsumexp(y_raw)  # normalize log probs in log space
        return -np.sum(y * y_raw_norm, axis=1)

    def gradient(self, y, y_raw):
        """
        Input
            y: 2d array of one-hot-encoded labels, shape=(no. examples, no. classes).
            y_hat: 2d array of pre-activation values, shape=(no. examples, no. classes).

        Returns 2d array of gradients w.r.t. the prediction; shape=(no. examples, no. classes).
        """
        y_hat = softmax(y_raw)
        return y_hat - y

    def hessian(self, y, y_raw):
        """
        Input
            y: 2d array of one-hot-encoded labels, shape=(no. examples, no. classes).
            y_hat: 2d array of pre-activation values, shape=(no. examples, no. classes).

        Returns 1d array of second-order gradients w.r.t. the prediction; shape=(no. examples, no. classes).
        """
        n_classes = y.shape[1]
        factor = n_classes / (n_classes - 1)  # rescaling redundant class

        y_hat = softmax(y_raw)
        return y_hat * (1 - y_hat) * factor

    def third(self, y, y_raw):
        """
        Input
            y: 2d array of one-hot-encoded labels, shape=(no. examples, no. classes).
            y_hat: 2d array of pre-activation values, shape=(no. examples, no. classes).

        Returns 2d array of third-order gradients w.r.t. the prediction; shape=(no. examples, no. classses).
        """
        n_classes = y.shape[1]
        factor = n_classes / (n_classes - 1)  # rescaling redundant class

        y_hat = softmax(y_raw)
        return y_hat * (1 - y_hat) * (1 - 2 * y_hat) * factor
