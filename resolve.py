import argparse
import numpy as np
from skimage import io
from scipy.optimize import minimize


# helper functions
def get_coords(height, width):
    coords = []
    for h in range(height):
        for w in range(width):
            coords.append([h, w])
    return np.array(coords)


def get_normalized_coords(height, width):
    coords = []
    height, width = np.linspace(0, 1, height), np.linspace(0, 1, width)
    for h in height:
        for w in width:
            coords.append([h, w])
    return np.array(coords)


def normalize(image):
    return 0.5 * (2 * (image-np.min(image))/(np.max(image)-np.min(image)) - 1)


def get_parameters(params, K):
    """ Gets the estimated parameters from results

    :param res:
    :param K: the number of images used
    """
    shifts = np.array(params[:2*K]).reshape(-1, 2)
    angles = np.array(params[2*K:2*K+K])
    gamma = params[-1]
    return shifts, angles, gamma


# Building Blocks
def cov(x_i, x_j, r=1.0, a=0.04):
    """ 
    The covariance function that determines smoothness of GP
    """
    x_i_sum = np.sum(x_i ** 2, axis=1).reshape(-1, 1)
    x_j_sum = np.sum(x_j ** 2, axis=1)
    dist = x_i_sum + x_j_sum - (2 * np.dot(x_i, x_j.T))
    out = a * np.exp(-(dist / r ** 2))
    return out


def transform_mat(x_i, x_j, center, shift, angle, gamma=2):
    """ This is the transformation matrix that converts a high
    resolution image into a low resolution image
    In the paper this is denoted using W_ji
    """
    x_i_sum = np.sum(x_i ** 2, axis=1).reshape(-1, 1)
    u_j = psf_center(x_j, center, shift, angle)
    u_j_sum = np.sum(u_j ** 2, axis=1)
    dist = x_i_sum + u_j_sum - (2 * np.dot(x_i, u_j.T))
    out = np.exp(-(dist / gamma ** 2))
    out /= np.sum(out, axis=0)
    return out.T


def psf_center(x_j, center, shift, angle):
    """ Used to calculate the center of the psf
    This is the vector u_j in the paper
    """
    rotation_matrix = [
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]]
    rotation_matrix = np.array(rotation_matrix)

    u = np.dot(rotation_matrix, (x_j - center).T)
    u = u.T + center + shift
    return u


def variance(Z_x, W_K, beta):
    Z_x_inv = np.linalg.inv(Z_x)
    a = 0
    for w in W_K:
        a += np.dot(w.T, w)
    sigma = Z_x_inv + (beta * a)
    return np.linalg.inv(sigma)


def mean(W_K, Y_K, sigma, beta):
    a = 0
    for w, y in zip(W_K, Y_K):
        y = y.reshape(-1, 1)
        a += np.dot(w.T, y)
    mu = beta * np.dot(sigma, a)
    return mu


def marginal_log_likelihood(Z_x, W_K, Y_K, beta, M, K):
    """ Calculates the marginal log likehood"""
    sigma = variance(Z_x, W_K, beta)
    mu = mean(W_K, Y_K, sigma, beta)

    likelihood = 0
    for w, y in zip(W_K, Y_K):
        y = y.reshape(-1, 1)
        y_sum = np.sum(y ** 2, axis=0)
        w_u = np.dot(w, mu)
        w_u_sum = np.sum(w_u ** 2, axis=0)
        likelihood += y_sum + w_u_sum - (2 * np.dot(y.flatten(), w_u.flatten()))
    likelihood *= beta
    likelihood += np.dot(np.dot(mu.T, np.linalg.inv(Z_x)), mu)[0][0]
    Z_x_sign, Z_x_logdet = np.linalg.slogdet(Z_x)
    sigma_sign, sigma_logdet = np.linalg.slogdet(sigma)
    likelihood += (Z_x_sign * Z_x_logdet) - (sigma_sign * sigma_logdet) - (K * M * np.log(beta))
    likelihood *= -0.5
    return likelihood


def compute_nll(X_n, X_m, Y_K, center, shifts, angles, beta, gamma):
    """ Computes the negative marginal log likelihood """
    Z_x = cov(X_n, X_n) + beta * np.eye(len(X_n))

    W_K = [transform_mat(X_n, X_m, center, shift, angle, gamma) for shift, angle in zip(shifts, angles)]
    W_K = np.array(W_K)

    K = len(shifts)
    M = len(X_m)
    nll = -marginal_log_likelihood(Z_x, W_K, Y_K, beta=beta, M=M, K=K)
    return nll


def compute_nll_theta(theta, X_n, X_m, Y_K, center, beta):
    """ Computes the negative marginal log likelihood """
    K = len(Y_K)
    M = len(X_m)

    Z_x = cov(X_n, X_n) + beta * np.eye(len(X_n))

    shifts = np.array(theta[:2*K]).reshape(-1, 2)
    angles = np.array(theta[2*K:2*K+K])
    gamma = theta[-1]

    W_K = [transform_mat(X_n, X_m, center, shift, angle, gamma) for shift, angle in zip(shifts, angles)]
    W_K = np.array(W_K)
    nll = -marginal_log_likelihood(Z_x, W_K, Y_K, beta=beta, M=M, K=K)
    return nll


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path',
                        type=str,
                        required=True,
                        help='The path to the image')
    parser.add_argument('--num-images',
                        type=int,
                        required=False,
                        default=16,
                        help='The number of low resolution images to use')
    args = parser.parse_args()
    image_path = args.image_path
    num_images = args.num_images

    image = io.imread(image_path)

    theta = np.zeros((num_images*3+1,))
    theta[-1] = 4  # gamma

    X_m = get_normalized_coords(9, 9)
    X_n = get_normalized_coords(18, 18)

    # set initial image and params
    beta = 0.05 ** 2
    center = np.array([40, 40])
    shifts = [[0, 0]]  # store for comparison
    angles = [0]  # store for comparison
    Y_K = [normalize(image[300:309, 400:409].flatten())]

    for _ in range(num_images-1):
        xshift = np.random.randint(-2, 3)
        yshift = np.random.randint(-2, 3)
        shifts.append([xshift, yshift])
        angles.append(0)
        i = normalize(image[300+yshift:309+yshift, 400+xshift:409+xshift].flatten())
        Y_K.append(i)
    Y_K = np.array(Y_K)

    print(shifts)
    print(angles)

    res = minimize(compute_nll_theta, theta,
                   args=(X_n, X_m, Y_K, center, beta),
                   method='CG')
    print(res.success)
    print(res.message)
    shifts, angles, gamma = get_parameters(res.x, num_images)

    print(shifts)
    print(angles)
    print(gamma)
