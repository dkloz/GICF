"""File that defines instance similarity. In this case, instances are sentences."""

import numpy as np


def get_sim_matrix(X, t='rbf', var=0.5):
    """X is a matrix of embeddings of size NxD (D is embedding Dimension). This function returns a matrix of size NxN
    that has the similarity between vector i and j in the position i,j. Similarity can be either cosine or an
    rbf kernel with some variance"""
    if t == 'rbf':
        return get_matrix_rbf(X, var)
    elif t == 'cosine' or t == 'cos':
        return get_matrix_cos(X)
    else:
        print'Wrong option for similarity. Choices are "rfb" and "cos"'


def get_matrix_rbf(X, var, have_var=True):
    """e^-||(x_i - x_j)||/ 2*var^2 RBF"""
    (rows, _) = X.shape
    K = np.empty((rows, rows))
    for i in range(rows):
        c = np.sum(np.abs(X - X[i]) ** 2, axis=-1) ** (1. / 2)  # norm
        if (have_var):
            c = c / (2 * (var ** 2))
        K[i] = c
    w = np.exp(- K.T)

    return w


def get_matrix_cos(X):
    """https://en.wikipedia.org/wiki/Cosine_similarity"""
    similarity = np.dot(X, X.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine
