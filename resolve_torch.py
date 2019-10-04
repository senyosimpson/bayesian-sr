import argparse
import torch
import numpy as np
from PIL import Image
from skimage import io
from skimage import transform
from skimage import color


# helper functions
def get_coords(height, width):
    coords = []
    for h in range(height):
        for w in range(width):
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


def generate_lr_images(hr_image, num_images, borders, var, gamma=2, return_all=True):
    x_start, x_end, y_start, y_end = borders
    shifts = [(0, 0)]  # store for comparison
    shifts.extend([(np.random.randint(-2, 3), np.random.randint(-2, 3)) for _ in range(num_images - 1)])
    angles = [0]  # store for comparison
    angles.extend([0 for _ in range(num_images - 1)])  # angle = np.random.randint(-4, 5)
    Y_K = []

    image = np.asarray(hr_image)
    image = image.flatten().reshape(-1, 1)
    image = normalize(image)

    w, h = hr_image.size
    w_down, h_down = w//4, h//4
    center_h, center_w = int(np.ceil(h_down/2)), int(np.ceil(w_down/2))
    center = np.array([center_h, center_w])

    X_n = get_coords(h, w)
    X_m = get_coords(h_down, w_down)

    for idx, (shift, angle) in enumerate(zip(shifts, angles)):
        print('Generating low resolution image %d/%d' % (idx+1, num_images))
        trans_mat = transform_mat(X_n, X_m, center, shift, angle, gamma=gamma)
        Y_k = np.dot(trans_mat, image).reshape(h_down, w_down)
        Y_k = Y_k[center_h-y_start:center_h+y_end, center_w-x_start:center_w+x_end].flatten()
        Y_k += var * np.random.randn(*Y_k.shape)
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


def transform_mat(x_i, x_j, center, shift, angle, gamma=2, upscale_factor=4):
    """ This is the transformation matrix that converts a high
    resolution image into a low resolution image
    In the paper this is denoted using W_ji
    """
    x_i_sum = torch.sum(x_i ** 2, dim=1)
    u_j = psf_center(x_j, center, shift, angle, upscale_factor)
    u_j_sum = torch.sum(u_j ** 2, dim=1).reshape(-1, 1)
    dist = x_i_sum + u_j_sum - (2 * torch.mm(u_j, x_i.t()))
    out = torch.exp(-(dist / gamma ** 2))
    out = out / torch.sum(out, dim=1).reshape(-1, 1)
    return out


def psf_center(x_j, center, shift, angle, upscale_factor=4):
    """ Used to calculate the center of the psf
    This is the vector u_j in the paper
    """
    #angle = np.radians(angle)
    rotation_matrix = [
        [torch.cos(angle), torch.sin(angle)],
        [-torch.sin(angle), torch.cos(angle)]]
    rotation_matrix = torch.tensor(rotation_matrix).to(torch.device('cuda'))

    u = torch.mm(rotation_matrix, (x_j - center).t())
    u = u.t() + center  # + shift
    if upscale_factor:
        u = u * upscale_factor
    u = u + shift
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


def compute_nll(shifts, gamma, angles, X_n, X_m, Y_K, center, beta, upscale_factor=4):
    """ Computes the negative marginal log likelihood """
    K = len(Y_K)
    M = len(X_m)
    var = 1 / beta

    Z_x = cov(X_n, X_n) + var * torch.eye(len(X_n)).to(torch.device('cuda'))

    # if angles are not given, fetch them from theta
    #if angles is None:
    #    angles = theta[2*K:2*K+K]
    # if gamma is not given, fetch it from theta

    W_K = [transform_mat(X_n, X_m, center, shift, angle, upscale_factor=upscale_factor, gamma=gamma)
           for shift, angle in zip(shifts, angles)]
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
    parser.add_argument('--max-iters',
                        type=int,
                        default=5,
                        required=False,
                        help='Maximum number of iterations for optimizer')
    parser.add_argument('--upscale-factor',
                        type=int,
                        required=False,
                        default=4,
                        help='the upscaling factor')
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
    max_iters = args.max_iters
    upscale_factor = args.upscale_factor
    seed = args.seed

    if seed:
        np.random.seed(seed)
        #torch.manual_seed(seed)

    hr_image = io.imread(image_path)
    hr_image = color.rgb2gray(hr_image)
    hr_image = transform.resize(hr_image, (200, 200), anti_aliasing=True)

    # set initial image and params
    beta = 1 / (0.05 ** 2)
    var = 1 / beta
    beta = torch.tensor(beta).to(device)
    var = torch.tensor(var).to(device)
    gamma = torch.tensor(2.).to(device)
    shifts = [torch.tensor([0., 0.]).to(device)]  # store for comparison
    angles = [torch.tensor(0.).to(device)]  # store for comparison
    Y_K = []

    # generate LR images
    print('Generating reference LR image')
    image = np.asarray(hr_image)
    image = torch.tensor(image).to(device)
    image = image.flatten().reshape(-1, 1)
    image = normalize(image)

    h, w = hr_image.shape
    w_down, h_down = w//4, h//4
    center_h, center_w = float(np.ceil(h_down/2)), float(np.ceil(w_down/2))
    center = torch.tensor([center_h, center_w]).to(device)
    center_h_down, center_w_down = int(np.ceil(h_down/2)), int(np.ceil(w_down/2))

    X_n = get_coords(h, w).to(device)
    X_m = get_coords(h_down, w_down)
    X_m = X_m.to(device)

    shift = torch.tensor([0., 0.]).to(device)
    angle = torch.tensor(0.).to(device)
    trans_mat = transform_mat(X_n, X_m, center, shift, angle, gamma=gamma).to(device)
    Y_k = torch.mm(trans_mat, image).reshape(h_down, w_down).to(device)
    Y_k = Y_k[center_h_down-4:center_h_down+5, center_w_down-4:center_w_down+5].flatten()
    Y_k += var * torch.randn(*Y_k.shape).to(device)
    Y_K.append(normalize(Y_k))

    for idx in range(num_images - 1):
        print('Generating low resolution image %d/%d' % (idx+1, num_images-1))
        xshift = float(np.random.randint(-2, 3))
        yshift = float(np.random.randint(-2, 3))
        shift = torch.tensor([xshift, yshift]).to(device)
        shifts.append(shift)
        #angle = 8 * np.random.random_sample() - 4
        #angle = float(np.random.randint(-4, 5))
        angle = torch.tensor(0.).to(device)
        #angle = np.random.randint(-4, 5)
        angles.append(angle)
        trans_mat = transform_mat(X_n, X_m, center, shift, angle, gamma=gamma).to(device)
        Y_k = torch.mm(trans_mat, image).reshape(h_down, w_down)
        Y_k = Y_k[center_h_down-4:center_h_down+5, center_w_down-4:center_w_down+5].flatten()
        Y_k += var * torch.randn(*Y_k.shape).to(device)
        Y_K.append(normalize(Y_k))

    print('Starting parameter estimation')
    X_n_shape = (36, 36)
    X_m_shape = (9, 9)
    X_n = get_coords(*X_n_shape).to(device)
    X_m = get_coords(*X_m_shape).to(device)
    center = torch.tensor([5., 5.]).to(device)

    init_guess_shifts = torch.zeros((num_images, 2)).to(device)
    init_guess_angles = torch.zeros(num_images).to(device)
    init_guess_gamma = torch.tensor(4.).to(device)

    print('Estimating shift parameters')
    init_guess_shifts.requires_grad = True
    #init_guess_gamma.requires_grad = True
    optimizer = torch.optim.Adam([init_guess_shifts], lr=0.005)
    losses = []
    num_steps = 1500
    for i in range(num_steps):
        print('Step %d/%d' % (i+1, num_steps))
        optimizer.zero_grad()
        loss = compute_nll(init_guess_shifts, init_guess_gamma, angles, X_n, X_m, Y_K, center, beta, 4)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Current loss %.5f' % loss.item())

    # log results
    estimated_shifts = init_guess_shifts
    for shift, est_shift in zip(shifts, estimated_shifts):
        print('Original Shift:', shift)
        print('Estimated Shift:', est_shift)
        print()

    # print('Estimating shift and angle parameters')
    # theta = np.concatenate((estimated_shifts, init_guess_angles))
    # res = minimize(compute_nll, theta,
    #                args=(X_n, X_m, Y_K, center, beta, init_guess_gamma),
    #                method='CG',
    #                options=options)
    # params = res.x
    # estimated_shifts = params[:2*num_images]
    # estimated_angles = params[2*num_images:2*num_images+num_images]

    print('Refining shift parameters and estimating PSF width parameter')
    init_guess_gamma.requires_grad = True
    optimizer = torch.optim.Adam([estimated_shifts, init_guess_gamma], lr=0.005)
    num_steps = 1500
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = compute_nll(estimated_shifts, init_guess_gamma, angles, X_n, X_m, Y_K, center, beta, 4)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # log results
    estimated_gamma = init_guess_gamma
    for shift, est_shift in zip(shifts, estimated_shifts):
        print('Original Shift:', shift)
        print('Estimated Shift:', est_shift)
        print()

    print('Original Gamma:', gamma)
    print('Estimated Gamma:', estimated_gamma)

    print('Estimating high resolution image')
