import argparse
import numpy as np
from PIL import Image
from scipy.optimize import minimize
import skimage.io as skio


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
    """ Gets the estimated parameters from results """
    shifts = np.array(params[:2*K]).reshape(-1, 2)
    angles = np.array(params[2*K:2*K+K])
    gamma = params[-1]
    return shifts, angles, gamma


# building blocks
def cov(x_i, x_j, r=1.0, a=0.04):
    """
    The covariance function that determines smoothness of GP
    """
    x_i_sum = np.sum(x_i ** 2, axis=1).reshape(-1, 1)
    x_j_sum = np.sum(x_j ** 2, axis=1)
    dist = x_i_sum + x_j_sum - (2 * np.dot(x_i, x_j.T))
    out = a * np.exp(-(dist / r ** 2))
    return out


def transform_mat(x_i, x_j, center, shift, angle, gamma=2, upscale_factor=4):
    """ This is the transformation matrix that converts a high
    resolution image into a low resolution image
    In the paper this is denoted using W_ji
    """
    x_i_sum = np.sum(x_i ** 2, axis=1)
    u_j = psf_center(x_j, center, shift, angle, upscale_factor)
    u_j_sum = np.sum(u_j ** 2, axis=1).reshape(-1, 1)
    dist = x_i_sum + u_j_sum - (2 * np.dot(u_j, x_i.T))
    out = np.exp(-(dist / gamma ** 2))
    out /= np.sum(out, axis=1).reshape(-1, 1)
    return out


def psf_center(x_j, center, shift, angle, upscale_factor=4):
    """ Used to calculate the center of the psf
    This is the vector u_j in the paper
    """
    angle = np.radians(angle)
    rotation_matrix = [
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]]
    rotation_matrix = np.array(rotation_matrix)

    u = np.dot(rotation_matrix, (x_j - center).T)
    u = u.T + center
    if upscale_factor:
        u = upscale_factor * u
    u = u + shift
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
    """ Calculates the marginal log likelihood"""
    sigma = variance(Z_x, W_K, beta)
    mu = mean(W_K, Y_K, sigma, beta)

    likelihood = 0
    for w, y in zip(W_K, Y_K):
        y = y.reshape(-1, 1)
        y_sum = np.sum(y ** 2, axis=0)
        w_u = np.dot(w, mu)
        w_u_sum = np.sum(w_u ** 2, axis=0)
        likelihood += y_sum + w_u_sum - (2 * np.dot(y.flatten(), w_u.flatten()))
        likelihood = likelihood[0]
    likelihood *= beta
    likelihood += np.dot(np.dot(mu.T, np.linalg.inv(Z_x)), mu)[0][0]
    Z_x_sign, Z_x_logdet = np.linalg.slogdet(Z_x)
    sigma_sign, sigma_logdet = np.linalg.slogdet(sigma)
    likelihood += (Z_x_sign * Z_x_logdet) - (sigma_sign * sigma_logdet) - (K * M * np.log(beta))
    likelihood *= -0.5
    return likelihood


def compute_nll(theta, X_n, X_m, Y_K, center, beta, upscale_factor=4, gamma=None, angles=None):
    """ Computes the negative marginal log likelihood """
    K = len(Y_K)
    M = len(X_m)
    var = 1 / beta

    Z_x = cov(X_n, X_n) + var * np.eye(len(X_n))

    shifts = np.array(theta[:2*K]).reshape(-1, 2)
    # if angles are not given, fetch them from theta
    if angles is None:
        angles = np.array(theta[2*K:2*K+K])
    # if gamma is not given, fetch it from theta
    if gamma is None:
        gamma = theta[-1]

    W_K = [transform_mat(X_n, X_m, center, shift, angle, upscale_factor=upscale_factor, gamma=gamma)
           for shift, angle in zip(shifts, angles)]
    W_K = np.array(W_K)
    nll = -marginal_log_likelihood(Z_x, W_K, Y_K, beta=beta, M=M, K=K)
    return nll


def compute_posterior(x, X_n, X_m, Y_K, beta, shifts, angles, gamma, upscale_factor=4):
    M = len(Y_K[0])
    var = 1 / beta

    W_K = [transform_mat(X_n, X_m, center, shift, angle, upscale_factor=upscale_factor, gamma=gamma)
           for shift, angle in zip(shifts, angles)]
    W_K = np.array(W_K)

    Z_x = cov(X_n, X_n) + var * np.eye(len(X_n))
    Z_x_sign, Z_x_logdet = np.linalg.slogdet(Z_x)
    k = len(W_K[0])
    prior = (Z_x_sign * Z_x_logdet) + np.dot(np.dot(x.T, np.linalg.inv(Z_x)), x) + (k * np.log(2*np.pi))
    prior = -0.5 * prior

    likelihood = 0
    for w, y in zip(W_K, Y_K):
        y = y.reshape(-1, 1)
        y_sum = np.sum(y ** 2, axis=0)
        w_x = np.dot(w, x)
        w_x_sum = np.sum(w_x ** 2, axis=0)
        likelihood += y_sum + w_x_sum - (2 * np.dot(y.flatten(), w_x.flatten()))
        likelihood = likelihood[0]
    likelihood *= beta
    likelihood -= M * np.log(beta/(2*np.pi))
    likelihood *= -0.5
    posterior = prior + likelihood
    return posterior


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
    parser.add_argument('--max-iters',
                        type=int,
                        default=5,
                        required=False,
                        help='Maximum number of iterations for optimizer')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        required=False,
                        help='Random seed to make random functions deterministic')
    args = parser.parse_args()
    image_path = args.image_path
    num_images = args.num_images
    max_iters = args.max_iters
    seed = args.seed

    if seed:
        np.random.seed(seed)

    hr_image = Image.open(image_path).convert('L')
    # downsample image to make it faster to compute
    hr_image = hr_image.resize((300, 300), resample=Image.BICUBIC)

    # set initial image and params
    beta = 1 / (0.05 ** 2)
    var = 1 / beta
    gamma = 2
    shifts = [[0, 0]]  # store for comparison
    angles = [0]  # store for comparison
    Y_K = []

    # generate LR images
    print('Generating reference LR image')
    image = np.asarray(hr_image)
    image = image.flatten().reshape(-1, 1)
    image = normalize(image)

    w, h = hr_image.size
    w_down, h_down = w//4, h//4
    center = np.array([h_down//2, w_down//2])
    center_h, center_w = center

    X_n = get_coords(h, w)
    X_m = get_coords(h_down, w_down)

    trans_mat = transform_mat(X_n, X_m, center, [0, 0], 0, gamma=gamma)
    Y_k = np.dot(trans_mat, image).reshape(h_down, w_down)
    Y_k = Y_k[center_h-4:center_h+5, center_w-4:center_w+5].flatten()
    Y_k += var * np.random.randn(*Y_k.shape)
    Y_K.append(normalize(Y_k))

    for idx in range(num_images - 1):
        print('Generating low resolution image %d/%d' % (idx+1, num_images-1))
        xshift = np.random.randint(-2, 3)
        yshift = np.random.randint(-2, 3)
        shifts.append([xshift, yshift])
        #angle = 8 * np.random.random_sample() - 4
        angle = 0
        #angle = np.random.randint(-4, 5)
        angles.append(angle)
        trans_mat = transform_mat(X_n, X_m, center, [xshift, yshift], angle, gamma=gamma)
        Y_k = np.dot(trans_mat, image).reshape(h_down, w_down)
        Y_k = Y_k[center_h-4:center_h+5, center_w-4:center_w+5].flatten()
        Y_k += var * np.random.randn(*Y_k.shape)
        Y_K.append(normalize(Y_k))

    print('Starting parameter estimation')
    X_n_shape = (50, 50)
    X_m_shape = (9, 9)
    X_n = get_coords(*X_n_shape)
    X_m = get_coords(*X_m_shape)
    center = np.array([5, 5])
    init_guess_shifts = np.zeros((num_images*2))
    init_guess_angles = np.zeros(num_images)
    init_guess_gamma = 4
    shift_bounds = [(-2, 2) for _ in range(num_images*2)]
    gamma_bounds = (1, None)
    options = {
        'disp': True,
        'maxiter': max_iters
    }

    print('Estimating shift parameters')
    theta = init_guess_shifts
    res = minimize(compute_nll, theta,
                   args=(X_n, X_m, Y_K, center, beta, 4, init_guess_gamma, init_guess_angles),
                   method='CG',
                   options=options)
    params = res.x
    estimated_shifts = params[:2*num_images]

    # print('Estimating shift and angle parameters')
    # theta = np.concatenate((estimated_shifts, init_guess_angles))
    # res = minimize(compute_nll, theta,
    #                args=(X_n, X_m, Y_K, center, beta, init_guess_gamma),
    #                method='CG',
    #                options=options)
    # params = res.x
    # estimated_shifts = params[:2*num_images]
    # estimated_angles = params[2*num_images:2*num_images+num_images]

    print('Estimating shift, angle and PSF width parameters')
    theta = np.concatenate((estimated_shifts, [init_guess_gamma]))
    bounds = np.concatenate((shift_bounds, [gamma_bounds]))
    res = minimize(compute_nll, theta,
                   args=(X_n, X_m, Y_K, center, beta, 4, None, init_guess_angles),
                   method='CG',
                   options=options)
    params = res.x
    estimated_shifts = np.array(params[:2*num_images]).reshape(-1, 2)
    # estimated_angles = np.array(params[2*num_images:2*num_images+num_images])
    estimated_gamma = params[-1]

    print('Original Shifts:', shifts)
    print('Estimated Shifts:', estimated_shifts)
    # print('Original Angles:', angles)
    # print('Estimated Angles:', estimated_angles)
    print('Original Gamma:', gamma)
    print('Estimated Gamma:', estimated_gamma)

    print('Estimating high resolution image')
    X_n = get_coords(h, w)
    X_m = get_coords(h_down, w_down)
    x = np.random.rand(h, w)
    x = 0.5 * (2 * x - 1)
    
    res = minimize(compute_posterior, theta,
                   args=(x, X_n, X_m, Y_K, beta, shifts, init_guess_angles, gamma),
                   method='CG',
                   options=options)

    x = res.x
    x = np.array(x).reshape(*X_n_shape)
    skio.imsave('artifacts/out.png', x)
