import warnings

import numpy as np


def softmax(x, axis=None):
    max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=axis, keepdims=True)
    f_x = e_x / sum
    return f_x


def sigmoid(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return 1.0 / (1.0 + np.exp(-x))
