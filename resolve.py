import numpy as np


def cov(x_i, x_j, r=1.0, a=1.0):
    """ 
    The covariance function that determines smoothness of GP
    """
    x_i_sum = np.sum(x_i**2, axis=1).reshape(-1, 1)
    x_j_sum = np.sum(x_j**2, axis=1)
    dist = x_i_sum + x_j_sum - (2 * np.dot(x_i, x_j.T))
    out = a * np.exp(-(dist / r**2))
    return out


def transform_mat(x_i, x_j, center, shift, angle, gamma=2):
    """ This is the transformation matrix that converts a high
    resolution image into a low resolution image
    In the paper this is denoted using W_ji
    """
    x_i_sum = np.sum(x_i**2, axis=1).reshape(-1, 1)
    u_j = psf_center(x_j, center, shift, angle)
    u_j_sum = np.sum(u_j**2, axis=1)
    dist = x_i_sum + u_j_sum - (2 * np.dot(x_i, u_j.T))
    out = np.exp(-(dist / gamma**2))
    out /= np.sum(out, axis=0)
    return out


def psf_center(x_j, center, shift, angle):
    """ Used to calculate the center of the psf
    This is the vector u_j in the paper
    """
    center = center.reshape(-1, 2)
    shift = shift.reshape(-1, 2)
    rotation_matrix = [
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]]
    if isinstance(angle, np.ndarray):
        rotation_matrix = []
        for a in angle:
            rotation_matrix.append([
                [np.cos(a), np.sin(a)],
                [-np.sin(a), np.cos(a)]])
        rotation_matrix = np.array(rotation_matrix)

    u = np.dot(rotation_matrix, (x_j - center).T)
    u = u.T + center + shift
    return u


def variance(Z_x, W_K, W_ji, beta):
    Z_x_inv = np.linalg.inv(Z_x)
    #d = 0
    #for k in range(len(W)):
    #    d += np.dot(W_vec.T, W[k])
    a = np.dot(W_K.T, W_ji)
    a = np.sum(a, axis=0)
    sigma = Z_x_inv + (beta * a)
    return np.linalg.inv(sigma)


def mean(W_K, y, beta, sigma):
    """
    :param W_K:
    :param y: a kxM dimension matrix
    :param beta:
    :param sigma:
    :return:
    """
    #a = 0
    #for k in range(len(y)):
    #    a += np.dot(WK.T, y[k])
    a = np.dot(W_K.T, y)
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


# gamma is a learnable parameter
def compute_likelihood(X, y, center, shifts, gamma):
    # takes in the necessary parameters to calculate everything and return a likelihood score
    pass