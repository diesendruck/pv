
# Private Support Points: Regression Examples

import argparse
import collections
import json
import logging
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import pandas as pd
import pdb
import scipy.stats as stats
import sklearn
import time

from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sp_utils import (
    energy,
    get_energy_sensitivity,
    get_support_points,
    sample_sp_exp_mech,
    scale_01)

from kme_utils import sample_kme_synthetic, weighted_mmd


# Set up command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--new_results', dest='new_results', action='store_true')
args = vars(parser.parse_args())
new_results = args['new_results']


# Set up logging.
run_logfile = '../results/regression_logs/run.log'
if os.path.isfile(run_logfile):
    os.remove(run_logfile)
# Without following lines, log file was not created.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=run_logfile,
                             filemode='a',
                             format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                             datefmt='%H:%M:%S',
                             level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING)
_LOG = logging.getLogger('[comparison]')


# Separate log file for result objects.
results_logfile = '../results/regression_logs/results.log'
if new_results:
    try:
        os.remove(results_logfile)
    except OSError:
        pass


def sorted_dict(d):
    return collections.OrderedDict(sorted(d.items()))


def test_regression(data, data_heldout):
    """Computes regression accuracy, averaged over multiple runs.
    
    Each run computes accuracy with k-fold cross validation.
    
    Args:
        data (array): Training data.
        data_heldout (array): Testing/heldout data.
    
    Returns:
        result (scalar): MSE value on test data.
    """
    
    def regress(X_train, X_test, Y_train, Y_test, polynomial=False):
        if polynomial:
            poly = PolynomialFeatures(degree=3)
            X_train = poly.fit_transform(X_train)
            X_test = poly.fit_transform(X_test)
            
        # Fits linear model given a train-test split.
        # Returns MSE value of fitted model.
        lm = LinearRegression()
        lm.fit(X_train, Y_train)

        Y_pred = lm.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)

        return mse
        
    result = regress(data[:, :-1], data_heldout[:, :-1],
                     data[:, -1], data_heldout[:, -1], polynomial=True)
    
    return result


def visualize_data(data, title=None):
    # Visualize data with pairs plot.
    if len(data) > 500:
        data = data[np.random.choice(len(data), 500)]             
    graph = pd.plotting.scatter_matrix(pd.DataFrame(data), figsize=(10, 10));
    if title:
        plt.suptitle(title)
    plt.show()


def resample_from_histdd(H, edge_sets, n=100, plot=False):
    """Resamples data set from histogram, uniformly over bins.
    
    Args:
        H (array): Arrays of counts per bin.
        edge_sets (array): Arrays of edge boundaries per dim.
        n (int): Number of points to sample.

    Returns:
        resampled (array): Newly sampled data.
    """
    bin_widths = [np.diff(edges)[0] for edges in edge_sets]
    midpoints = [edges[:-1] + np.diff(edges) / 2 for edges in edge_sets]

    # Compute CDF of counts, then normalize.
    cdf = np.cumsum(H.ravel())
    cdf = cdf / cdf[-1]
    
    # Sample uniform, associate to CDF values.
    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)
    
    # Fetch associated indices from original grid.
    unraveled_shape = [len(r) for r in midpoints]
    hist_indices = np.array(np.unravel_index(value_bins, unraveled_shape))
    
    # Sample uniformly on bin.
    random_from_cdf = []
    num_dims = len(hist_indices)
    for i in range(num_dims):
        bin_width_i = bin_widths[i]
        mids_i = midpoints[i][hist_indices[i]]
        vals_i = mids_i + np.random.uniform(low=-bin_width_i / 2,
                                            high=bin_width_i / 2,
                                            size=mids_i.shape)
        random_from_cdf.append(vals_i)
    resampled = np.array(random_from_cdf).T
    #random_from_cdf = np.array([midpoints[i][hist_indices[i]] for i in range(num_dims)])
    
    # Visualize data.
    if plot:
        visualize_data(resampled, 'resampled')
        
    return resampled
    
    
def run_experiments(dataset_name, data_orig, num_supp_list, alphas,
                    max_iter, lr, num_sp_samples, results_logfile=None,
                    method='mh', burnin=5000, num_cv_splits=None, power=None):
    """Runs panel of experiments for different number of support points
    and different alpha settings.
    
    Args:
        dataset_name: String name.
        data_orig: NumPy array of data. NOTE: Target var must be last column [!].
        num_supp_list: List of support point set sizes.
        alphas: List of alphas.
        max_iter: Int, number of steps in support point optimization.
        lr: Float, learning rate of support point optimization.
        num_sp_samples: Int, number of Support Point sets over which 
          to average regression performance.
        method: String, name of private-sampling method. ['diffusion', 'mh']
        burnin: Int, number of samples to burn in MH sampler.
        num_cv_splits: Int, number of cross validation splits.
        power: Int, power in energy metric. [1, 2]
      
    Returns:
        None
    """
    
    # Fetch, scale, and shuffle data.
    data_scaled = data_orig
    assert np.min(data_scaled) >= 0 and np.max(data_scaled) <= 1, 'Scaling incorrect.'
    np.random.shuffle(data_scaled)

    # Create folds for cross validation.
    kf = KFold(n_splits=num_cv_splits)

    # Do an experiment run for each train-test split.
    for train_index, test_index in kf.split(data_scaled):
        print('Starting new data split.')
        _LOG.info('Starting new data split.')
        
        if max(max(train_index), max(test_index)) >= len(data_scaled):
            pdb.set_trace()
        data = data_scaled[train_index]
        data_heldout = data_scaled[test_index]
    
        visualize_data(data, title='data')
    
        # --------------------------------------
        # Test regression on FULL TRAINING data.
        # --------------------------------------
        
        mse = test_regression(data, data_heldout)
        result = {
                'dataset_name': dataset_name,
                'mse': mse,
                'tag': 'full_training'} 
        with open(results_logfile, 'a') as f:
            json.dump(sorted_dict(result), f)
            f.write(os.linesep)


        # --------------------------------------------------
        # Test regression on RANDOM SUBSETS (size num_supp).
        # --------------------------------------------------

        for num_supp in num_supp_list:

            random_subsets_data = [
                data[np.random.choice(len(data), num_supp)] for
                _ in range(num_sp_samples)]

            for subset in random_subsets_data:
                mse = test_regression(subset, data_heldout)
                result = {
                        'dataset_name': dataset_name,
                        'mse': mse,
                        'size': num_supp, 
                        'tag': 'random_subset'}
                with open(results_logfile, 'a') as f:
                    json.dump(sorted_dict(result), f)
                    f.write(os.linesep)


        # -------------------------------------------------------------
        # Test regression on REPEATED SP optimizations (size num_supp).
        # -------------------------------------------------------------

        for num_supp in num_supp_list:

            _LOG.info('Starting repeated SP optimizations, num_supp={}.'.format(num_supp))
            
            sp_sets = []
            for _ in range(num_sp_samples):
                y_opt, e_opt = get_support_points(data, num_supp, max_iter, lr,
                                                  is_tf=True, Y_INIT_OPTION='uniform',
                                                  clip='data', plot=False)
                sp_sets.append(y_opt)
            
            # Show an example of support points.
            visualize_data(y_opt, title='SP, num_supp={}'.format(num_supp))
            
            for sp_set in sp_sets:
                mse = test_regression(sp_set, data_heldout)
                result = {
                        'dataset_name': dataset_name,
                        'mse': mse,
                        'size': num_supp, 
                        'tag': 'support_points'}
                with open(results_logfile, 'a') as f:
                    json.dump(sorted_dict(result), f)
                    f.write(os.linesep)


            e_rand = np.mean([energy(random_subsets_data[j], data, power=power)[0] for
                j in range(len(random_subsets_data))])
            e_supp = np.mean([energy(sp_sets[j], data, power=power)[0] for
                j in range(len(sp_sets))])
            print('num_supp={}, e_rand={:.8f}, e_supp={:.8f}'.format(num_supp, e_rand, e_supp))


        # --------------------------------------------
        # Test regression on PERTURBED HISTOGRAM data.
        # --------------------------------------------
        
        _LOG.info('Starting perturbed histograms.')

        for alpha in alphas:
            _N, _DIM = data.shape
            num_bins = (_N * alpha / 10) ** ((2 * _DIM) / (2 + _DIM))
            num_bins_per_dim = min(20, int(np.round(num_bins ** (1 / _DIM))))  # https://arxiv.org/pdf/1504.05998.pdf
            try:
                # Get true histogram, and perturb with Laplace noise.
                H, edges = np.histogramdd(data, bins=num_bins_per_dim)
            except:
                print('[!] Perturbed histogram does not fit in memory. Skipping.')
                _LOG.warning(('Skipping histogram for dataset={}, alpha={}. '
                              'num_bins_per_dim ** dim, {} ** {} is too large.').format(
                    dataset_name, alpha, num_bins_per_dim, _DIM))
                break
            
            for _ in range(num_sp_samples):
                # Perturb histogram counts with Laplace noise.
                H_perturbed = H + np.random.laplace(loc=0, scale=1/alpha, size=H.shape)

                # Resample from perturbed histogram, using uniform sampling per bin.
                perturbed_hist = resample_from_histdd(H_perturbed, edges, n=_N)

                # Evaluate regression performance on heldout data.
                mse = test_regression(perturbed_hist, data_heldout)
                result = {
                        'dataset_name': dataset_name,
                        'mse': mse,
                        'alpha': alpha, 
                        'tag': 'perturbed_hist'}
                with open(results_logfile, 'a') as f:
                    json.dump(sorted_dict(result), f)
                    f.write(os.linesep)


        # ----------------------------------------------------------
        # Test regression on PRIVATE SUPPORT POINTS (size num_supp).
        # ----------------------------------------------------------

        for num_supp in num_supp_list:

            # For each alpha, compute private support points and test regression on them.
            for alpha in alphas:

                print('Starting private SP, num_supp={}, alpha={}'.format(num_supp, alpha))
                _LOG.info('Starting private SP, num_supp={}, alpha={}'.format(num_supp, alpha))

                # Compute private support points.
                # For this (_n, alpha) combo, run sampler many times.
                priv_sp_sets = []
                for _ in range(num_sp_samples):
                    (private_sp,
                     energy_) = sample_sp_exp_mech(data, num_supp, alpha=alpha,
                                                   save_dir='../results/regression_logs')

                    priv_sp_sets.append(private_sp)
                

                # Test regression on private support points.
                for priv_sp_set in priv_sp_sets:
                    mse = test_regression(priv_sp_set, data_heldout)
                    result = {
                            'dataset_name': dataset_name,
                            'mse': mse,
                            'alpha': alpha,
                            'size': num_supp, 
                            'tag': 'private_support_points'}
                    with open(results_logfile, 'a') as f:
                        json.dump(sorted_dict(result), f)
                        f.write(os.linesep)


        # ------------------------------------------------------
        # Test regression on PRIVATE KME POINTS (size num_supp).
        # ------------------------------------------------------

        for num_supp in num_supp_list:

            # For each alpha, compute private support points and test regression on them.
            for alpha in alphas:

                print('Starting private KME, num_supp={}, alpha={}'.format(num_supp, alpha))
                _LOG.info('Starting private KME, num_supp={}, alpha={}'.format(num_supp, alpha))

                # Compute private KME points.
                # For this (_n, alpha) combo, run sampler many times.
                priv_kme_sets = []
                for _ in range(num_sp_samples):
                    private_kme, weights_ = sample_kme_synthetic(
                             data, M=num_supp, epsilon=alpha,
                             save_dir='../results/regression_logs',
                             plot=False)
                    priv_kme_sets.append(private_kme)

                # Test regression on private support points.
                for priv_kme_set in priv_kme_sets:
                    mse = test_regression(priv_kme_set, data_heldout)
                    result = {
                            'dataset_name': dataset_name,
                            'mse': mse,
                            'alpha': alpha,
                            'size': num_supp, 
                            'tag': 'private_kme'}
                    with open(results_logfile, 'a') as f:
                        json.dump(sorted_dict(result), f)
                        f.write(os.linesep)


def dict_filter_tag(rows, query, dataset, tag):
    """Gets results. Note: As saved, ordered by increasing N."""
    r = [d[query] for d in rows if
         d['dataset_name'] == dataset and
         d['tag'] == tag]
    return r

def dict_filter_alpha(rows, query, dataset, alpha):
    """Gets results. Note: As saved, ordered by increasing N."""
    r = [d[query] for d in rows if
         d['dataset_name'] == dataset and
         d['tag'] == 'private_support_points' and
         d['alpha'] == alpha]
    return r


def plot_final_results(results_logfile, dataset_name, x_ticks, alphas):
    """Plots regression results.
    
    Args:
        results_logfile (string): Name of results log file.
        dataset_name (string): Name of dataset, e.g. 'boston'.
        x_ticks (list/np array): Percentage sizes of sp sets.
        alphas (list): List of alphas used in experiment panel.
    
    Returns:
        None
    """

    # Load results.
    with open(results_logfile) as f:
        results = [json.loads(line) for line in f]
    results = [d for d in results if d['dataset_name'] == dataset_name]

    # Fetch sizes and set up x-axis for set sizes.
    sizes = np.unique([d['size'] for d in results if d['tag'] == 'random_subset'])
    x = np.array(x_ticks)
    x_jitter = (max(x) - min(x)) / 40.
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 8))
    fig.suptitle(dataset_name)
    
    # -------------------------------------
    # Make boxplot for full-size sets.
    # -------------------------------------
    
    _full_training = [d['mse'] for d in results if d['tag'] == 'full_training']
    
    # Perturbed histogram results.
    _hist_results = []
    for a in alphas:
        _hist_a = [
            d['mse'] for d in results if
            d['tag'] == 'perturbed_hist' and
            d['alpha'] == a]
        _hist_results.append(_hist_a)

    # Fetch non-empty results for boxplot.
    if len(_hist_results[0]) == 0:
        boxplot_contents = _full_training
        boxplot_labels = ['train']
    else:
        boxplot_contents = [_full_training] + _hist_results
        boxplot_labels = ['train'] + ['hist:{}'.format(a) for a in alphas]

    ax1.boxplot(boxplot_contents, labels=boxplot_labels)


    # ------------------------------------------------------------
    # Then make line plot with error bars for variably-sized runs.
    # ------------------------------------------------------------
    
    # Random subset.
    mses = []
    stds = []
    for size in sizes:
        res = [
            d['mse'] for d in results if
            d['tag'] == 'random_subset' and
            d['size'] == size]
        mses.append(np.mean(res))
        stds.append(np.std(res))
   
    ax2.errorbar(x, mses, yerr=stds, label='random_subset')

    # Support points.
    mses = []
    stds = []
    for size in sizes:
        res = [
            d['mse'] for d in results if
            d['tag'] == 'support_points' and
            d['size'] == size]
        mses.append(np.mean(res))
        stds.append(np.std(res))
    
    ax2.errorbar(x + 1 * x_jitter, mses, yerr=stds, label='support_points')
        
    # Private support points.
    for i, alpha in enumerate(alphas):
        mses = []
        stds = []
        for size in sizes:
            res = [d['mse'] for d in results if
                   d['tag'] == 'private_support_points' and
                   d['size'] == size and
                   d['alpha'] == alpha]
            
            mses.append(np.mean(res))
            stds.append(np.std(res))

        ax2.errorbar(x + 2 * x_jitter + i * x_jitter,
                     mses,
                     yerr=stds,
                     label=r'sp, $\alpha={}$'.format(alpha))

    # Private KME synthetic.
    for i, alpha in enumerate(alphas):
        mses = []
        stds = []
        for size in sizes:
            res = [d['mse'] for d in results if
                   d['tag'] == 'private_kme' and
                   d['size'] == size and
                   d['alpha'] == alpha]
            
            mses.append(np.mean(res))
            stds.append(np.std(res))

        ax2.errorbar(x + 3 * x_jitter + i * x_jitter,
                     mses,
                     yerr=stds,
                     label=r'kme, $\alpha={}$'.format(alpha))

    # ------------------------------------    
    
    ax1.set_xlabel('Data used for fitting')    
    ax2.set_xlabel('Number of Points, Fraction of Whole')
    ax1.set_ylabel('Mean Squared Error')

    ax2.legend()
    plt.savefig('../results/regression_logs/global_results_{}'.format(dataset_name))
    plt.show()
    plt.close()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# Full Experiments.
# To shorten run: alphas, sp_samples, cv_splits, burnin, thinning, for i in range(num_samples), subsetting count, subsetting dims.

run_boston = 1
run_diabetes = 1
run_california = 0

alphas = [10 ** p for p in [3, 4, 5]]  # [2, 3, 4, 5], [5, 4, 3, 2]
num_sp_samples = 100  # 10
num_cv_splits = 5  # 5
power = 1


################################
# BOSTON

# Get Boston data.
dataset_name = 'boston'
dataset = load_boston()

data = np.concatenate((dataset.data, dataset.target.reshape(-1, 1)), axis=1)
data = scale_01(data)
np.random.shuffle(data)

# Optional subsetting for troubleshooting.
#data = data[np.random.choice(len(data), 200, replace=False)]
data = data[:, [4, 5, 6, 13]]

# Define set of parameters to test.
train_size = int(len(data) * (num_cv_splits - 1) / num_cv_splits)
percent_num_supp = [0.02, 0.05, 0.1]
num_supp_list = [int(train_size * i) for i in percent_num_supp]
max_iter = 301 #501  # 301
lr = 0.5 #0.01         # 0.5
burnin = 5000

# TODO: PLOT ONLY ONCE RESULTS ARE IN.
#plot_final_results(results_logfile, 'boston', percent_num_supp, alphas)
#pdb.set_trace()

# Run experiments and print results.
if run_boston:
    run_experiments(dataset_name, data, num_supp_list, alphas, max_iter, lr,
            num_sp_samples, results_logfile=results_logfile, burnin=burnin,
            num_cv_splits=num_cv_splits, power=power)
if run_boston:
    plot_final_results(results_logfile, 'boston', percent_num_supp, alphas)


################################
# DIABETES

# Get diabetes data.
dataset_name = 'diabetes'
dataset = load_diabetes()

data = np.concatenate((dataset.data, dataset.target.reshape(-1, 1)), axis=1)
data = scale_01(data)
np.random.shuffle(data)

# Optional subsetting for troubleshooting.
#data = data[np.random.choice(len(data), 100, replace=False)]
data = data[:, [4, 6, 7, 10]]

# Define set of parameters to test.
train_size = int(len(data) * (num_cv_splits - 1) / num_cv_splits) 
percent_num_supp = [0.02, 0.05, 0.1]
num_supp_list = [int(train_size * i) for i in percent_num_supp]
max_iter = 301
lr = 0.5
burnin = 5000

# Run experiments and print results.
if run_diabetes:
    run_experiments(dataset_name, data, num_supp_list, alphas, max_iter, lr,
            num_sp_samples, results_logfile=results_logfile,
            num_cv_splits=num_cv_splits, power=power)
if run_diabetes:
    plot_final_results(results_logfile, 'diabetes', percent_num_supp, alphas)


################################
# CALIFORNIA

# Get diabetes data.
dataset_name = 'california'
dataset = fetch_california_housing()

data = np.concatenate((dataset.data, dataset.target.reshape(-1, 1)), axis=1)
data = scale_01(data)
np.random.shuffle(data)

# Optional subsetting for troubleshooting.
data = data[np.random.choice(len(data), 500, replace=False)]
data = data[:, [1, 6, 7, 8]]

# Define set of parameters to test.
train_size = int(len(data) * (num_cv_splits - 1) / num_cv_splits) 
percent_num_supp = [0.02, 0.05, 0.1]
num_supp_list = [int(train_size * i) for i in percent_num_supp]
max_iter = 501
lr = 1.
burnin = 5000

# Run experiments and print results.
if run_california:
    run_experiments(dataset_name, data, num_supp_list, alphas, max_iter, lr,
            num_sp_samples, results_logfile=results_logfile,
            num_cv_splits=num_cv_splits, power=power)
if run_california:
    plot_final_results(results_logfile, 'california', percent_num_supp, alphas)
