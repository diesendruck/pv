import argparse
import numpy as np
from utils import json_dump
import pdb
import os

#np.random.seed(2)


"""
def generate_data_mixture_of_Gaussians(N_private, D):
    C = 10
    mu_means = 100.0 * np.ones(D)
    sigma_means = 200.0
    sigma_data = 30.0
    mus = np.random.normal(loc=mu_means, scale=sigma_means, size=(C, D))
    ws = np.array([1.0 / n for n in range(1, C+1)])
    ws /= np.sum(ws)
    memberships = np.random.choice(range(C), size=N_private, p=ws)
    X_private = []
    for n in range(N_private):
        mu = mus[memberships[n]]
        sample = np.random.normal(loc=mu, scale=sigma_data)
        X_private.append(sample)
    X_private = np.array(X_private)
    return X_private
"""


def generate_data_mixture_of_Gaussians_01(M_private, D, C, SIGMA, do_weighted=True):
    """Generates multimodal data set bounded to [0, 1].
    """
    centers_low = 0 + 4. * SIGMA
    centers_high = 1 - 4. * SIGMA
    mus = np.random.uniform(low=centers_low, high=centers_high, size=(C, D))

    if do_weighted:
        weights = np.array([1.0 / n for n in range(1, C + 1)])
        weights /= np.sum(weights)
        memberships = np.random.choice(range(C), size=M_private, p=weights)
    else:
        weights = np.array([1.0 for n in range(1, C + 1)])
        memberships = np.random.choice(range(C), size=M_private)
    X_private = []
    for n in range(M_private):
        mu = mus[memberships[n]]
        sample = np.random.normal(loc=mu, scale=SIGMA)
        X_private.append(sample)
    X_private = np.array(X_private)
    return X_private, weights


def load_balog_data(M, DIM, C, SIGMA, make_new=False, do_weighted=False):
    """Wrapper to make and save data, given number of points, 
       dimension, num clusters, and cluster variance.
    """
    # np.random.seed(0)
    DATA_PATH = '/Users/swilliamson/Work/Mo/pv/mv_sp_privacy/src/data/mixture_of_Gaussians_M{}_D{}_C{}_SIG{}.npz'. format(M, DIM, C, SIGMA)
    
    if not os.path.exists(DATA_PATH) or make_new:
        # Generate the private data
        x, weights = generate_data_mixture_of_Gaussians_01(M, DIM, C, SIGMA,
                                                           do_weighted=do_weighted)
        
        assert (M, DIM) == np.shape(x), 'Balog data dims do not match global params.'
    
        # Save the generated data
        dataset_name = 'mixture_of_Gaussians'
        path_save = '/Users/swilliamson/Work/Mo/pv/mv_sp_privacy/src/data/{}_M{}_D{}_C{}_SIG{}'.format(
            dataset_name, M, DIM, C, SIGMA)
        np.savez(path_save + '.npz', X_private=x)
        json_dump({'dataset': dataset_name,
                   'M': M,
                   'DIM': DIM,
                   'C': C,
                   'SIG': SIGMA,
                   'WEIGHTS': weights,
                  },
                  path_save + '.json')
    else:
        # Load existing.
        data = np.load(DATA_PATH)
        x = data['X_private']

    assert (M, DIM) == np.shape(x), 'Balog data dims do not match global params.'
    print('Loaded M={}, DIM={}, SIG={}, C={}'.format(M, DIM, SIGMA, C))
    
    return x


"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='Number of data points to generate')
    parser.add_argument('D', type=int, help='Dimensionality of generated data')
    parser.add_argument('C', type=int, help='Number of clusters')
    args_dict = vars(parser.parse_args())
    main(args_dict)
"""
