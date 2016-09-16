"""File with helper functions for array manipulation of the formulas written in the paper.
Each function has a wrapper that calls the function that calculates it, because of different things tried in experiments
"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_y(x, theta, classifier='logistic'):
    if classifier == 'logistic':
        return sigmoid(np.dot(x, theta))


def similarity_derivative(y, y_der_matrix, W_ij, classifier='logistic'):

    if classifier == 'logistic':
        return similarity_derivative_logistic(y, y_der_matrix, W_ij)


def similarity_derivative_logistic(y, y_der_matrix, W_ij):  # fine
    """Calculates the value of the derivative of the first term in the cost function that has to do with similarity.
    Returns an array"""
    y_diff = get_diff(y)
    a = np.dot(2 * np.multiply(W_ij, y_diff).T, y_der_matrix)  # .T so that the additions make sense
    b = np.dot(2 * np.multiply(W_ij, y_diff), y_der_matrix)
    return np.sum(a, axis=0) - np.sum(b, axis=0)


def get_diff(y):
    """2*(y - y.T) used for derivative"""
    y = y.reshape((-1, 1))
    d = np.subtract(y, y.T)
    return 2 * d


def calculate_y_der(y, x, classifier='logistic'):
    if classifier == 'logistic':
        return logistic_derivative(y, x)


def logistic_derivative(y, x):
    """returns the value derivative of the logistic function for a specific array of values"""
    y_der = np.multiply(y, (1 - y))
    a = y_der.reshape((-1, 1))
    return a * x


def group_derivative(y, y_der, gs, gl, fn='avg'):
    if fn == 'avg':
        return group_derivative_avg(y, y_der, gs, gl)


def group_derivative_avg(y, y_der, group_scores, group_lengths):
    """Calculates the value of the derivative of the second term in the cost function, which is based on the group
    cost. Again returns array"""
    (_, columns) = y_der.shape
    sum_array = np.zeros(columns)

    for i in range(len(group_scores)):
        if group_lengths[i] <= 0:  # if length > 0
            continue
        group_score = group_scores[i]
        frm = np.sum(group_lengths[0:i])
        to = frm + group_lengths[i]
        a = 2 * (np.average(y[frm:to]) - group_score * 2 * 0.5)
        b = (np.sum(y_der[frm:to, :], axis=0)) / group_lengths[i]
        sum_array += np.multiply(a, b)
    return sum_array
