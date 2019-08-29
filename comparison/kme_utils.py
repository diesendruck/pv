import argparse
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import pdb
import torch
dtype = torch.DoubleTensor
import torch.optim

from sklearn.linear_model import Lasso
from torch.autograd import Variable

from DP import Gaussian_mechanism
from kernels import EQKernel, compute_mean_feature_vector
from utils import json_load, json_dump, dedup_consecutive


def weighted_mmd(synthetic, data, kernel=None, K_xx_mean=None, weights=None):
    # Compute real distance from private embedding

    # Provide default uniform weights.
    n = len(synthetic)
    if weights is None:
        weights = np.array([1. / n for _ in range(n)])

    # Provide default kernel.
    if kernel is None:
        lengthscale = 0.5
        N, D = np.shape(data)
        gamma = 1.0 / lengthscale ** 2
        kernel = EQKernel(D, gamma)

    # Provide data-data portion of kernel.
    if K_xx_mean is None:
        K_xx_mean = kernel.get_kernel_matrix_mean(data)

    # Compute metric.
    K_zz = kernel.get_kernel_matrix(synthetic)
    K_zx_rmeans = kernel.get_kernel_matrix_rowmeans(synthetic, data)
    dist2 = (
        weights.dot(K_zz.dot(weights)) -
        2.0 * weights.dot(K_zx_rmeans) +
        K_xx_mean)
    dist_k_opt = np.sqrt(dist2)

    return dist_k_opt


def sample_kme_synthetic(
        X_private, M=None, epsilon=None, J=10000, num_iters=100,
        lr=0.5, lasso_alpha=10., lengthscale=0.5, uniform_weights=False,
        plot=False, save_dir=None):
    """Computes synthetic set via private kernel mean embeddings.

    Args:
      X_private (array): Data.
      M (int): Number of synthetic points in a single set.
      J (int): Number of random features in Fourier features projection.
      num_iters (int): Number of steps in greedy synthetic point optimization.
      lr (float): Learning rate of synthetic pt optim.
      lasso_alpha (float): Regularization parameter for weight optim.
      lengthscale (float): Kernel bandwidth for MMD.
      uniform_weights (boolean): Use uniform weights in synthetic pt optim.
      plot (boolean): To plot or not to plot.
      save_dir (string): Directory for plots.

    Returns:
      Z (numpy array): Synthetic points.
      weights (numpy array): Associated weights.
    """

    # Parameters / Initialization.
    N, D = np.shape(X_private)
    alpha = lasso_alpha / (2*N*J)   # Lasso regression regularization for w's
    delta = 1. / N
    Z_initial = X_public = np.random.uniform(size=(M, D))
    gamma = 1.0 / lengthscale ** 2
    kernel = EQKernel(D, gamma)

    # Compute kernel matrix mean K_xx_mean
    K_xx_mean = kernel.get_kernel_matrix_mean(X_private)

    # Get random feature functions.
    rf_np, rf_torch = kernel.get_random_features(J)

    # Compute random features of private data
    empirical_private = compute_mean_feature_vector(X_private, rf_np, J)

    # Add privatizing noise
    L2_sensitivity = 2.0 / N
    empirical_public = empirical_private + \
        Gaussian_mechanism(J, L2_sensitivity, epsilon, delta)

    # Run reduced set method "iterated approximate pre-images" (iteratively
    # construct M synthetic points) to approximate empirical_public
    Z = np.zeros((M, D))
    betas = []
    Psi = empirical_public
    for m in range(M):
        # Initialize and optimise location z with Torch
        zTorch = Variable(torch.from_numpy(Z_initial[m]),
                          requires_grad=True)
        PsiTorch = Variable(torch.from_numpy(Psi), requires_grad=False)
        optimizer = torch.optim.Adam([zTorch], lr=lr)
        for i in range(num_iters):
            optimizer.zero_grad()
            phiz = rf_torch(zTorch.view(1, D)).view(-1)  # -1 = J or 2*J
            loss = - PsiTorch.dot(phiz) ** 2 / phiz.dot(phiz)
            # Backward pass and gradient step
            loss.backward()
            optimizer.step()

        # Compute optimal weight beta and update residual vector
        z = zTorch.data.numpy()
        phiz = rf_np(z)
        if uniform_weights:
            beta = 1. / M
        else:
            beta = Psi.dot(phiz) / phiz.dot(phiz)
        Psi = Psi - beta * phiz

        # Save found synthetic point and its optimal weight
        Z[m] = z
        betas.append(beta)

    # Compute optimal reweighting
    if uniform_weights:
        weights = np.array([1. / M for _ in range(M)])
    else:
        # Re-fit weights with a regularization.
        Phi = rf_np(Z)
        clf = Lasso(fit_intercept=False, alpha=alpha)
        clf.fit(np.transpose(Phi), empirical_public)
        weights = np.reshape(clf.coef_, (m+1))  # Below, reshape needed for case m=0

    # Compute weighted distance between synthetic and data.
    dist_k_opt = weighted_mmd(Z, X_private, kernel=kernel, K_xx_mean=K_xx_mean,
            weights=weights)
    print('[Result] epsilon={}, M={}, mmd = {}'.format(epsilon, M, dist_k_opt))

    # Optionally plot.
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(X_private[:, 0], X_private[:, 1], c='gray', alpha=0.3)
        ax.scatter(Z[:, 0], Z[:, 1], c=weights, cmap='cool')  # Plot Lasso weights.
        for j, txt in enumerate(weights):
            ax.annotate(round(txt, 2), (Z[j, 0], Z[j, 1]))
        ax.set_title('num_data={}, num_supp={}, eps={}\nd_kme={:.5f}'.format(
            N, M, epsilon, dist_k_opt))
        plt.savefig(os.path.join(save_dir, 'balog_eps{}.png'.format(epsilon)))
        plt.close()

    #print(weights)
    #print(weights == 0)
    return Z, weights
