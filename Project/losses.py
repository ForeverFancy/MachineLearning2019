import numpy as np

def mean_squared_error(y_pred, y_true):
    '''
    error = 1/(2N) * (y_pred - y_true)^2
    dy = (y_pred - y_true)
    '''
    error = 1 / 2 * np.mean(np.sum(np.square(y_pred - y_true), axis=-1))
    dy = y_pred - y_true
    return error, dy

def cross_entropy_error(y_pred, y_true):
    # TODO: Check it.
    y_shift = y_pred - np.max(y_pred, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    dy = y_probability - y_true
    return loss, dy
