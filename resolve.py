# Define functions to calculate all the various matrices
# Most of the calculations are done per image which means no multidimensional problem formulations
# For the W matrix, every pixel ji is normalized with the sum of all pixels in that row

import numpy as np


def cov(x1, x2, r=1.0, a=1.0):
    """ The covariance function that determines smoothness of GP
    It encodes some similarity about points near each other
    x1 and x2 are the indexes to those points, hence they are
    technically the same vector and the logic for batch calculations
    is done here instead of separate indexing of i,j coordinates
    x1 has shape m x d
    x2 has shape n x d
    out has shape m x n (cov matrix between all points)
    """
    _x1 = np.sum(x1**2, axis=1).reshape(-1, 1)
    _x2 = np.sum(x2**2, axis=1)
    dist = np.sqrt((_x1 + _x2 - (2 * np.dot(_x1, _x2.T))))
    out = a * np.exp(- (dist / r**2))
    return out


def transform_mat(x1, x2, gamma):
    """ This is the transformation matrix that converts a high
    resolution image into a low resolution image
    In the paper this is denoted using W
    Remember normalization
    """
    raise NotImplementedError


def psf_centre(center, shifts, angle=None, x=None):
    """ Used to calculate the center of the psf
    This is the vector u in the paper

    In the 1D case, we leave out the rotation
    """
    #rotation_matrix = [
    #    [np.cos(angle), np.sin(angle)],
    #    [-np.sin(angle), np.cos(angle)]]
    # u = np.dot(rotation_matrix, (x-center).T) + center + shifts (this should return a vector)
    u = center + shifts
    return u


def variance(Zx, W_vec, W, beta):
    Zx_inv = np.linalg.inv(Zx)
    d = 0
    for k in range(len(W)):
        d += np.dot(W_vec.T, W[k])
    sigma = Zx_inv + (beta * d)
    return np.linalg.inv(sigma)


def mean(beta, sigma, W_vec, y):
    d = 0
    for k in range(len(y)):
        d += np.dot(W_vec.T, y[k])
    mu = beta * sigma * d
    return mu


def log_marginal_likelihood(beta, y, W, mu, Zx, sigma, K, M):
    likelihood = 0
    for k in range(len(y)):
        _y = np.sum(y[k]**2, axis=1).reshape(-1, 1)
        _y_hat = np.sum(np.dot(W[k].T, mu) ** 2, axis=1)
        likelihood += _y + _y_hat - (2 * np.dot(_y, _y_hat.T))

    _u = np.dot(mu.T, np.linalg.inv(Zx))
    _u = np.dot(_u, mu)
    likelihood += _u
    likelihood += np.log(Zx) - np.log(sigma) - (K * M * np.log(beta))
