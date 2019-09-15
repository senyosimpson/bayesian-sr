# Define functions to calculate all the various matrices
# Most of the calculations are done per image which means no multidimensional problem formulations
# For the W matrix, every pixel ji is normalized with the sum of all pixels in that row

import numpy as np


def cov(x_i, x_j, r=1.0, a=1.0):
    """ The covariance function that determines smoothness of GP
    It encodes some similarity about points near each other
    x_i and x_j are the indexes to those points, hence they are
    technically the same vector and the logic for batch calculations
    is done here instead of separate indexing of i,j coordinates
    x_i has shape m x d
    x_j  has shape n x d
    out has shape m x n (cov matrix between all points)
    """
    x_i_sum = np.sum(x_i**2, axis=1).reshape(-1, 1)
    x_j_sum = np.sum(x_j**2, axis=1)
    dist = x_i_sum + x_j_sum - (2 * np.dot(x_i, x_j.T))
    out = a * np.exp(-(dist / r**2))
    return out


def transform_mat(x_i, x_j, center, shifts, gamma):
    """ This is the transformation matrix that converts a high
    resolution image into a low resolution image
    In the paper this is denoted using W_ji
    Remember normalization
    """
    x_i_sum = np.sum(x_i**2, axis=1).reshape(-1, 1)
    u_j = psf_center(x_j, center, shifts)
    u_j_sum = np.sum(u_j**2, axis=1)
    dist = x_i_sum + u_j_sum - (2 * np.dot(x_i, u_j.T))
    out = np.exp(-(dist / gamma**2))
    # below could be incorrect, remember to change to axis=1 if results are poor
    out /= np.sum(out, axis=0)
    return out


def psf_center(x_j, center, shifts, angle=None):
    """ Used to calculate the center of the psf
    This is the vector u_j in the paper

    shifts is a vector with shape (x,)

    In the 1D case, we leave out the rotation
    """
    rotation_matrix = [
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]]
    #u = np.dot(rotation_matrix, (x_j - center).T) + center + shifts
    u = x_j + shifts
    # in the 2d case, u = center + shifts.reshape(-1,2)
    return u


def variance(Z_x, W_vec, W, beta):
    Z_x_inv = np.linalg.inv(Z_x)
    #d = 0
    #for k in range(len(W)):
    #    d += np.dot(W_vec.T, W[k])
    a = np.dot(W_vec, W)
    a = np.sum(a, axis=0)
    sigma = Z_x_inv + (beta * a)
    return np.linalg.inv(sigma)


def mean(WK, y, beta, sigma):
    """
    :param WK:
    :param y: a kxM dimension matrix
    :param beta:
    :param sigma:
    :return:
    """
    #a = 0
    #for k in range(len(y)):
    #    a += np.dot(WK.T, y[k])
    a = np.dot(WK.T, y)
    a = np.sum(a, axis=0)
    mu = beta * np.matmul(sigma, a)
    return mu


def marginal_log_likelihood(beta, y, W, mu, Z_x, sigma, K, M):
    sigma = variance(Z_x, W, W, beta)  # arguments need to be changed here for the first W
    mu = mean(W, y, beta, sigma)

    y_sum = np.sum(y**2, axis=1)
    W_u = np.matmul(W, mu)
    W_u_sum = np.sum(W_u**2, axis=1)
    likelihood = y_sum + W_u_sum - (2 * np.dot(y_sum, W_u.T))
    likelihood *= beta

    likelihood += np.matmul(np.dot(mu.T, np.linalg.inv(Z_x)), mu)
    likelihood += np.log(np.linalg.det(Z_x)) - np.log(np.linalg.det(sigma)) - (K * M * np.log(beta))
    likelihood *= -0.5
    return likelihood


def compute_likelihood():
    # takes in the necessary parameters to calculate everything and return a likelihood score
    pass

