import argparse
import torch
import numpy as np
from skimage import io
from skimage import transform
from skimage import color


# helper functions
def get_coords(height, width, space=1):
    coords = []
    for h in range(0, height, space):
        for w in range(0, width, space):
            coords.append([float(h), float(w)])
    return torch.tensor(coords)


def get_normalized_coords(height, width):
    coords = []
    height, width = torch.linspace(0, 1, height), torch.linspace(0, 1, width)
    for h in height:
        for w in width:
            coords.append([h, w])
    return torch.tensor(coords)


def normalize(image):
    image = 0.5 * (2 * (image - torch.min(image)) / (torch.max(image) - torch.min(image)) - 1)
    image = image.float()
    return image


def generate_lr_images(hr_image, num_images, borders, var, gamma=2, shifts=None, angles=None, return_all=True):
    x_start, x_end, y_start, y_end = borders
    if shifts is None:
        shifts = [(0., 0.)]  # store for comparison
        shifts.extend([(float(np.random.randint(-2, 3)), float(np.random.randint(-2, 3))) for _ in range(num_images - 1)])
        shifts = torch.tensor(shifts).to(torch.device('cuda'))
    if angles is None:
        angles = [0.]  # store for comparison
        angles.extend([float(np.random.randint(-2, 3)) for _ in range(num_images - 1)])
        angles = torch.tensor(angles).to(torch.device('cuda'))
    Y_K = []

    image = np.asarray(hr_image)
    image = torch.tensor(image).to(torch.device('cuda'))
    image = image.flatten().reshape(-1, 1)
    image = normalize(image)

    h, w = hr_image.shape
    w_down, h_down = w//4, h//4
    center_h, center_w = float(np.ceil(h/2)), float(np.ceil(w/2))
    center = torch.tensor([center_h, center_w]).to(device)
    center_h_down, center_w_down = int(np.ceil(h_down/2)), int(np.ceil(w_down/2))

    X_n = get_coords(h, w).to(torch.device('cuda'))
    X_m = get_coords(h, w, space=4).to(torch.device('cuda'))

    for idx, (shift, angle) in enumerate(zip(shifts, angles)):
        print('Generating low resolution image %d/%d' % (idx+1, num_images))
        trans_mat = transform_mat(X_n, X_m, center, shift, angle, gamma=gamma).to(torch.device('cuda'))
        Y_k = torch.mm(trans_mat, image).reshape(h_down, w_down).to(torch.device('cuda'))
        Y_k = Y_k[center_h_down-y_start:center_h_down+y_end, center_w_down-x_start:center_w_down+x_end].flatten()
        Y_k += var * torch.randn(*Y_k.shape).to(torch.device('cuda'))
        Y_K.append(normalize(Y_k))

    if return_all:
        return Y_K, shifts, angles
    return Y_K


# building blocks
def cov(x_i, x_j, r=1.0, a=0.04):
    """
    The covariance function that determines smoothness of GP
    """
    x_i_sum = torch.sum(x_i ** 2, dim=1).reshape(-1, 1)
    x_j_sum = torch.sum(x_j ** 2, dim=1)
    dist = x_i_sum + x_j_sum - (2 * torch.mm(x_i, x_j.t()))
    out = a * torch.exp(-(dist / r ** 2))
    return out


def transform_mat(x_i, x_j, center, shift, angle, gamma=2):
    """ This is the transformation matrix that converts a high
    resolution image into a low resolution image
    In the paper this is denoted using W_ji
    """
    x_i_sum = torch.sum(x_i ** 2, dim=1)
    u_j = psf_center(x_j, center, shift, angle)
    u_j_sum = torch.sum(u_j ** 2, dim=1).reshape(-1, 1)
    dist = x_i_sum + u_j_sum - (2 * torch.mm(u_j, x_i.t()))
    out = torch.exp(-(dist / gamma ** 2))
    out = out / torch.sum(out, dim=1).reshape(-1, 1)
    return out


def psf_center(x_j, center, shift, angle):
    """ Used to calculate the center of the psf
    This is the vector u_j in the paper
    """
    angle = (np.pi / 180.) * angle
    rotation_matrix = torch.zeros(2, 2)
    rotation_matrix[0, 0] = torch.cos(angle)
    rotation_matrix[0, 1] = torch.sin(angle)
    rotation_matrix[1, 0] = -torch.sin(angle)
    rotation_matrix[1, 1] = torch.cos(angle)
    rotation_matrix = rotation_matrix.to(torch.device('cuda'))

    u = torch.mm(rotation_matrix, (x_j - center).t())
    u = u.t() + center + shift
    return u


def variance(Z_x, W_K, beta):
    Z_x_inv = torch.inverse(Z_x)
    a = torch.tensor(0.).to(torch.device('cuda'))
    for w in W_K:
        a = a + torch.mm(w.t(), w)
    sigma = Z_x_inv + (beta * a)
    return torch.inverse(sigma)


def mean(W_K, Y_K, sigma, beta):
    a = torch.tensor(0.).to(torch.device('cuda'))
    for w, y in zip(W_K, Y_K):
        y = y.reshape(-1, 1)
        a = a + torch.mm(w.t(), y)
    mu = beta * torch.mm(sigma, a)
    return mu


def marginal_log_likelihood(Z_x, W_K, Y_K, beta, M, K):
    """ Calculates the marginal log likelihood"""
    sigma = variance(Z_x, W_K, beta)
    mu = mean(W_K, Y_K, sigma, beta)

    likelihood = torch.tensor(0.).to(torch.device('cuda'))
    for w, y in zip(W_K, Y_K):
        y = y.reshape(-1, 1)
        y_sum = torch.sum(y ** 2, dim=0)
        w_mu = torch.mm(w, mu)
        w_mu_sum = torch.sum(w_mu ** 2, dim=0)
        likelihood = likelihood + (y_sum + w_mu_sum).squeeze() - (2 * torch.dot(y.flatten(), w_mu.flatten()))
    likelihood = likelihood * beta
    likelihood = likelihood + torch.mm(torch.mm(mu.t(), torch.inverse(Z_x)), mu).squeeze()
    likelihood = likelihood + torch.logdet(Z_x) - torch.logdet(sigma) - (K * M * torch.log(beta))
    likelihood = likelihood * -0.5
    return likelihood


def compute_nll(shifts, angles, gamma, X_n, X_m, Y_K, center, beta):
    """ Computes the negative marginal log likelihood """
    K = len(Y_K)
    M = len(X_m)
    var = 1 / beta

    Z_x = cov(X_n, X_n) + var * torch.eye(len(X_n)).to(torch.device('cuda'))

    W_K = [transform_mat(X_n, X_m, center, shift, angle, gamma=gamma)
           for shift, angle in zip(shifts, angles)]
    nll = -marginal_log_likelihood(Z_x, W_K, Y_K, beta=beta, M=M, K=K)
    return nll


def compute_posterior(x, X_n, X_m, Y_K, beta, center, shifts, angles, gamma):
    M = len(Y_K[0])
    var = 1 / beta

    W_K = [transform_mat(X_n, X_m, center, shift, angle, gamma=gamma)
           for shift, angle in zip(shifts, angles)]

    Z_x = cov(X_n, X_n) + var * torch.eye(len(X_n)).to(torch.device('cuda'))
    k = len(W_K[0])
    prior = torch.logdet(Z_x) + torch.mm(torch.mm(x.t(), torch.inverse(Z_x)), x)
    prior = prior + k * torch.log(torch.tensor(2 * np.pi).to(torch.device('cuda')))
    prior = -0.5 * prior

    likelihood = torch.tensor(0.).to(torch.device('cuda'))
    for w, y in zip(W_K, Y_K):
        y = y.reshape(-1, 1)
        y_sum = torch.sum(y ** 2, dim=0)
        w_x = torch.mm(w, x)
        w_x_sum = torch.sum(w_x ** 2, dim=0)
        likelihood = likelihood + (y_sum + w_x_sum).squeeze() - (2 * torch.dot(y.flatten(), w_x.flatten()))
    likelihood = beta * likelihood
    likelihood = likelihood - M * torch.log(beta / (2 * np.pi))
    likelihood = -0.5 * likelihood
    posterior = prior + likelihood
    posterior = -1 * posterior
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
    parser.add_argument('--upscale-factor',
                        type=int,
                        required=False,
                        default=4,
                        help='the image upscaling factor')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        required=False,
                        help='Random seed to make random functions deterministic')
    args = parser.parse_args()

    print('Use Cuda:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_path = args.image_path
    num_images = args.num_images
    upscale_factor = args.upscale_factor
    seed = args.seed

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

    hr_image = io.imread(image_path)
    hr_image = color.rgb2gray(hr_image)
    hr_image = transform.resize(hr_image, (100, 100), anti_aliasing=True)

    # Set params
    beta = 1 / (0.05 ** 2)
    var = 1 / beta
    beta = torch.tensor(beta).to(device)
    var = torch.tensor(var).to(device)
    gamma = torch.tensor(2.).to(device)

    # Get LR images
    Y_K, shifts, angles = generate_lr_images(hr_image, num_images, (7, 8, 7, 8), var, gamma)
    shifts = shifts.to(device)
    angles = angles.to(device)

    print('Starting parameter estimation')
    h, w = 75, 75
    h_down, w_down = 60, 60
    center_h, center_w = float(np.ceil(h/2)), float(np.ceil(w/2))
    center = torch.tensor([center_h, center_w]).to(device)

    X_n = get_coords(h, w).to(device)
    X_m = get_coords(h_down, w_down, space=4).to(device)

    init_guess_shifts = torch.zeros((num_images, 2)).to(device)
    init_guess_angles = torch.zeros(num_images).to(device)
    init_guess_gamma = torch.tensor(4.).to(device)

    print('Estimating shift parameters, angle parameters and PSF width parameter')
    init_guess_shifts.requires_grad = True
    init_guess_gamma.requires_grad = True
    init_guess_angles.requires_grad = True
    optimizer = torch.optim.Adam([init_guess_shifts, init_guess_angles, init_guess_gamma], lr=0.005)
    num_steps = 1000
    for i in range(num_steps):
        print('Step %d/%d' % (i+1, num_steps))
        optimizer.zero_grad()
        loss = compute_nll(init_guess_shifts, init_guess_angles, init_guess_gamma, X_n, X_m, Y_K, center, beta)
        loss.backward()
        optimizer.step()
        print('Current loss %.5f' % loss.item())

    init_guess_shifts.requires_grad = False
    init_guess_angles.requires_grad = False
    init_guess_gamma.requires_grad = False
    # log results
    estimated_shifts = init_guess_shifts
    estimated_angles = init_guess_angles
    estimated_gamma = init_guess_gamma
    for shift, est_shift, angle, est_angle in zip(shifts, estimated_shifts, angles, estimated_angles):
        print('Original Shift:', shift)
        print('Estimated Shift:', est_shift)
        print('Original Angle:', angle)
        print('Estimated Angle:', est_angle)
        print()
    print('Original Gamma:', gamma)
    print('Estimated Gamma:', estimated_gamma)

    print('Estimating high resolution image')
    # Generate new set of LR images with different dimensions but same shifts and angles
    Y_K = generate_lr_images(hr_image, num_images, (100, 100, 100, 100), var, gamma, shifts, angles, return_all=False)
    h, w = 100, 100
    h_down, w_down = 25, 25
    center_h, center_w = float(np.ceil(h/2)), float(np.ceil(w/2))
    center = torch.tensor([center_h, center_w]).to(device)

    X_n = get_coords(h, w).to(device)
    X_m = get_coords(h, w, space=4).to(device)

    est_hr_image = torch.zeros(h, w).reshape(-1, 1).to(device)
    est_hr_image.requires_grad = True
    optimizer = torch.optim.Adam([est_hr_image], lr=0.005)
    num_steps = 200
    for idx in range(num_steps):
        print('Step %d/%d' % (idx+1, num_steps))
        optimizer.zero_grad()
        loss = compute_posterior(est_hr_image, X_n, X_m, Y_K, beta, center,
                                 estimated_shifts,
                                 estimated_angles,
                                 estimated_gamma)
        loss.backward()
        optimizer.step()
        print('Current loss %.5f' % loss.item())

        if idx % 100 == 0:
            out_hr_image = est_hr_image.reshape(h, w)
            out_hr_image = out_hr_image.cpu().detach().numpy()
            io.imsave('/artifacts/out_%d.png' % idx, out_hr_image)

    out_hr_image = est_hr_image.reshape(h, w)
    out_hr_image = out_hr_image.cpu().detach().numpy()
    io.imsave('/artifacts/final_out.png', out_hr_image)
