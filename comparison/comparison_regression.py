
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
import tensorflow as tf
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


def test_regression(data, data_heldout, weights=None):
    """Computes regression accuracy, averaged over multiple runs.
    
    Each run computes accuracy with k-fold cross validation.
    
    Args:
        data (array): Training data.
        data_heldout (array): Testing/heldout data.
        weights (array): Weights for each data point.
    
    Returns:
        mse (scalar): MSE value on test data.
    """
    train_X = data[:, :-1]
    train_Y = data[:, -1]
    test_X = data_heldout[:, :-1]
    test_Y = data_heldout[:, -1]
    
    #if polynomial:
    #    poly = PolynomialFeatures(degree=3)
    #    train_X = poly.fit_transform(train_X)
    #    test_X = poly.fit_transform(test_X)
        
    # Fit (optionally weighted) linear model given a train-test split.
    if weights is not None:

        #######################################################
        # TODO: Consider scaling weights to be non-negative.

        # Optimize betas of regression with given weights.
        # argmin_b sum_i (w_i * (y_i - b^t.x_i) ** 2)
        n, d = train_X.shape
        train_Y = np.reshape(train_Y, [-1, 1])
        weights = np.reshape(weights, [-1, 1])
        assert len(weights) == n, 'Weights dim must match train_X'

        # Set up TensorFlow graph.
        tf.reset_default_graph()
        tf_weights = tf.placeholder(tf.float32, [None, 1], name='weights')
        tf_X = tf.placeholder(tf.float32, [None, d], name='X')
        tf_Y = tf.placeholder(tf.float32, [None, 1], name='Y')
        tf_coefs = tf.Variable(np.ones(shape=(d, 1)), name='coefs', dtype=tf.float32)
        tf_Y_hat = tf.matmul(tf_X, tf_coefs)
        tf_loss_pred = tf.reduce_sum(tf_weights * tf.square(tf_Y - tf_Y_hat))
        tf_coef_norm = tf.norm(tf_coefs, ord=1)
        tf_loss = tf_loss_pred + tf_coef_norm

        tf_optim = tf.train.GradientDescentOptimizer(1e-2).minimize(tf_loss)

        tf_init_op = tf.global_variables_initializer()
        tf_gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        tf_sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf_gpu_options)

        # Fit model.
        prediction_losses = []
        with tf.Session(config=tf_sess_config) as sess:
            sess.run(tf_init_op)
            for it in range(200):
                _, loss_, coefs_, loss_pred_, coef_norm_ = sess.run(
                        [tf_optim, tf_loss, tf_coefs, tf_loss_pred, tf_coef_norm],
                        {tf_X: train_X, tf_Y: train_Y, tf_weights: weights})
                if it % 10 == 0:
                    prediction_losses.append(loss_)
                    print('\n\nit {}. tf_loss = {},\ntf_coefs = {}\n'.format(it, loss_, coefs_))
                    print('loss_pred = {}, coef_norm = {}'.format(loss_pred_, coef_norm_))
                    y_, y_hat_ = sess.run(
                            [tf_Y, tf_Y_hat],
                            {tf_X: test_X[:5],
                             tf_Y: np.reshape(test_Y, [-1, 1])[:5]})
                    print('weights')
                    print(weights)
                    print('test_Y')
                    print(y_)
                    print('test_Y_hat')
                    print(y_hat_)

            # Measure MSE on test data.
        print(prediction_losses)
        pdb.set_trace()

        #######################################################
            

    else:
        lm = LinearRegression()
        lm.fit(train_X, train_Y)
        Y_pred = lm.predict(test_X)
        mse = mean_squared_error(test_Y, Y_pred)

    return mse
        

def visualize_data(data, title=None):
    # Visualize data with pairs plot.
    if len(data) > 500:
        data = data[np.random.choice(len(data), 500)]             
    graph = pd.plotting.scatter_matrix(pd.DataFrame(data), figsize=(10, 10));
    if title:
        plt.suptitle(title)
    plt.show()
    plt.savefig('../results/regression_logs/visualize_data_{}.png'.format(title))
    plt.close()


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
        print('\n\n\n---STARTING NEW DATA SPLIT---\n\n\n')
        _LOG.info('Starting new data split.')
        
        if max(max(train_index), max(test_index)) >= len(data_scaled):
            pdb.set_trace()
        data = data_scaled[train_index]
        data_heldout = data_scaled[test_index]
    
        visualize_data(data, title='train')
    
        """
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
                mmd = weighted_mmd(subset, data)
                result = {
                        'dataset_name': dataset_name,
                        'mse': mse,
                        'mmd': mmd,
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
            visualize_data(y_opt, title='SP, num_supp={}, e_opt={:.6f}'.format(
                num_supp, e_opt))
            
            for sp_set in sp_sets:
                mse = test_regression(sp_set, data_heldout)
                mmd = weighted_mmd(sp_set, data)
                result = {
                        'dataset_name': dataset_name,
                        'mse': mse,
                        'mmd': mmd,
                        'size': num_supp, 
                        'tag': 'support_points'}
                with open(results_logfile, 'a') as f:
                    json.dump(sorted_dict(result), f)
                    f.write(os.linesep)


            #e_rand = np.mean([energy(random_subsets_data[j], data, power=power)[0] for
            #    j in range(len(random_subsets_data))])
            #e_supp = np.mean([energy(sp_sets[j], data, power=power)[0] for
            #    j in range(len(sp_sets))])
            #print('num_supp={}, e_rand={:.8f}, e_supp={:.8f}'.format(num_supp, e_rand, e_supp))


        # --------------------------------------------
        # Test regression on PERTURBED HISTOGRAM data.
        # --------------------------------------------
        
        _LOG.info('Starting perturbed histograms.')

        for alpha in alphas:
            _N, _DIM = data.shape
            # TODO: What bin count (per dim) is too small?
            # https://arxiv.org/pdf/1504.05998.pdf
            num_bins = (_N * alpha / 10) ** ((2 * _DIM) / (2 + _DIM))
            num_bins_per_dim = min(10, int(np.round(num_bins ** (1 / _DIM))))  
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
                mmd = weighted_mmd(perturbed_hist, data)
                result = {
                        'dataset_name': dataset_name,
                        'mse': mse,
                        'mmd': mmd,
                        'alpha': alpha, 
                        'tag': 'perturbed_hist'}
                with open(results_logfile, 'a') as f:
                    json.dump(sorted_dict(result), f)
                    f.write(os.linesep)


        # ------------------------------------------
        # Test regression on PRIVATE SUPPORT POINTS.
        # ------------------------------------------

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
                                                   save_dir='../results/regression_logs',
                                                   plot=False)
                    priv_sp_sets.append(private_sp)
                

                # Test regression on private support points.
                for priv_sp_set in priv_sp_sets:
                    mse = test_regression(priv_sp_set, data_heldout)
                    mmd = weighted_mmd(priv_sp_set, data)
                    result = {
                            'dataset_name': dataset_name,
                            'mse': mse,
                            'mmd': mmd,
                            'alpha': alpha,
                            'size': num_supp, 
                            'tag': 'private_support_points'}
                    with open(results_logfile, 'a') as f:
                        json.dump(sorted_dict(result), f)
                        f.write(os.linesep)


        # ---------------------------------------
        # Test regression on PRIVATE KME UNIFORM.
        # ---------------------------------------

        for num_supp in num_supp_list:

            # For each alpha, compute private KME uniform points and test
            # regression.
            for alpha in alphas:

                print(('Starting private KME uniform, '
                       'num_supp={}, alpha={}').format(num_supp, alpha))
                _LOG.info(('Starting private KME uniform, '
                           'num_supp={}, alpha={}').format(num_supp, alpha))

                # Compute private KME uniform points.
                priv_kme_uniform_sets = []
                for _ in range(num_sp_samples):
                    private_kme_uniform, weights_ = sample_kme_synthetic(
                             data, M=num_supp, epsilon=alpha,
                             uniform_weights=True,  # Unweighted points.
                             save_dir='../results/regression_logs',
                             plot=False)
                    priv_kme_uniform_sets.append(private_kme_uniform)

                # Test regression on private KME uniform points.
                for priv_kme_uniform_set in priv_kme_uniform_sets:
                    mse = test_regression(priv_kme_uniform_set, data_heldout)
                    mmd = weighted_mmd(priv_kme_uniform_set, data)
                    result = {
                            'dataset_name': dataset_name,
                            'mse': mse,
                            'mmd': mmd,
                            'alpha': alpha,
                            'size': num_supp, 
                            'tag': 'private_kme_uniform'}
                    with open(results_logfile, 'a') as f:
                        json.dump(sorted_dict(result), f)
                        f.write(os.linesep)

        """

        # ----------------------------------------
        # Test regression on PRIVATE KME WEIGHTED. 
        # ----------------------------------------

        for num_supp in num_supp_list:

            # For each alpha, compute private KME weighted points and test
            # regression.
            for alpha in alphas:

                print(('Starting private KME weighted, '
                       'num_supp={}, alpha={}').format(num_supp, alpha))
                _LOG.info(('Starting private KME weighted, '
                           'num_supp={}, alpha={}').format(num_supp, alpha))

                # Compute private KME weight points.
                priv_kme_weighted_sets = []
                for _ in range(num_sp_samples):
                    points_, weights_ = sample_kme_synthetic(
                             data, M=num_supp, epsilon=alpha,
                             uniform_weights=False,  # Weighted points.
                             save_dir='../results/regression_logs',
                             plot=False)
                    priv_kme_weighted_sets.append([points_, weights_])

                # Test regression on private KME weighted points.
                for priv_kme_weighted_set in priv_kme_weighted_sets:
                    points_, weights_ = priv_kme_weighted_set
                    mse = test_regression(
                            points_, data_heldout, weights=weights_)
                    mmd = weighted_mmd(points_, data, weights=weights_)
                    result = {
                            'dataset_name': dataset_name,
                            'mse': mse,
                            'mmd': mmd,
                            'alpha': alpha,
                            'size': num_supp, 
                            'tag': 'private_kme_weighted'}
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

    ax1.set_ylim(0,1)
    ax1.boxplot(boxplot_contents, labels=boxplot_labels)


    # ------------------------------------------------------------
    # Then make line plot with error bars for variably-sized runs.
    # ------------------------------------------------------------
    
    """
    ###############
    # Random subset.
    mses = []
    stds = []
    pct25 = []
    pct50 = []
    pct75 = []
    for size in sizes:
        res = [
            d['mse'] for d in results if
            d['tag'] == 'random_subset' and
            d['size'] == size]
        mses.append(np.mean(res))
        stds.append(np.std(res))
        pct25.append(np.percentile(res, 25))
        pct50.append(np.percentile(res, 50))
        pct75.append(np.percentile(res, 75))
   
    #ax2.errorbar(x, mses, yerr=stds, label='random_subset')
    ax2.set_ylim(0,1)
    ax2.errorbar(x, pct50, yerr=[pct25, pct75], label='random_subset')
    """

    ###############
    # Support points.
    mses = []
    stds = []
    pct25 = []
    pct50 = []
    pct75 = []
    for size in sizes:
        res = [
            d['mse'] for d in results if
            d['tag'] == 'support_points' and
            d['size'] == size]
        mses.append(np.mean(res))
        stds.append(np.std(res))
        pct25.append(np.percentile(res, 25))
        pct50.append(np.percentile(res, 50))
        pct75.append(np.percentile(res, 75))
    
    ax2.set_ylim(0,1)
    ax2.errorbar(x + 1 * x_jitter, pct50, yerr=[pct25, pct75], label='support_points')
        
    ###############
    # Private support points.
    for i, alpha in enumerate(alphas):
        mses = []
        stds = []
        pct25 = []
        pct50 = []
        pct75 = []
        for size in sizes:
            res = [d['mse'] for d in results if
                   d['tag'] == 'private_support_points' and
                   d['size'] == size and
                   d['alpha'] == alpha]
            
            mses.append(np.mean(res))
            stds.append(np.std(res))
            pct25.append(np.percentile(res, 25))
            pct50.append(np.percentile(res, 50))
            pct75.append(np.percentile(res, 75))

        ax2.set_ylim(0,1)
        ax2.errorbar(x + 2 * x_jitter + i * x_jitter,
                     pct50,
                     yerr=[pct25, pct75],
                     label=r'sp, $\alpha={}$'.format(alpha))

    ###############
    # Private KME uniform synthetic.
    for i, alpha in enumerate(alphas):
        mses = []
        stds = []
        pct25 = []
        pct50 = []
        pct75 = []
        for size in sizes:
            res = [d['mse'] for d in results if
                   d['tag'] == 'private_kme_uniform' and
                   d['size'] == size and
                   d['alpha'] == alpha]
            
            mses.append(np.mean(res))
            stds.append(np.std(res))
            pct25.append(np.percentile(res, 25))
            pct50.append(np.percentile(res, 50))
            pct75.append(np.percentile(res, 75))

        ax2.set_ylim(0,1)
        ax2.errorbar(x + 3 * x_jitter + i * x_jitter,
                     pct50,
                     yerr=[pct25, pct75],
                     label=r'kme_uniform, $\alpha={}$'.format(alpha))


    ###############
    # Private KME weighted synthetic.
    for i, alpha in enumerate(alphas):
        mses = []
        stds = []
        pct25 = []
        pct50 = []
        pct75 = []
        for size in sizes:
            res = [d['mse'] for d in results if
                   d['tag'] == 'private_kme_weighted' and
                   d['size'] == size and
                   d['alpha'] == alpha]
            
            mses.append(np.mean(res))
            stds.append(np.std(res))
            pct25.append(np.percentile(res, 25))
            pct50.append(np.percentile(res, 50))
            pct75.append(np.percentile(res, 75))

        ax2.set_ylim(0,1)
        ax2.errorbar(x + 3 * x_jitter + i * x_jitter,
                     pct50,
                     yerr=[pct25, pct75],
                     label=r'kme_weighted, $\alpha={}$'.format(alpha))

    # ------------------------------------    
    
    ax1.set_xlabel('Data used for fitting')    
    ax2.set_xlabel('Number of Points, Fraction of Whole')
    ax1.set_ylabel('Mean Squared Error')

    ax2.legend()
    plt.savefig('../results/regression_logs/global_results_{}'.format(dataset_name))
    plt.show()
    plt.close()


def main():
    # Full Experiments.
    # To shorten run: alphas, sp_samples, cv_splits, burnin, thinning,
    #   for i in range(num_samples), subsetting count, subsetting dims.

    run_boston = 1
    run_diabetes = 1
    run_california = 1

    alphas = [10 ** p for p in [3, 4, 5]]  # [2, 3, 4, 5], [5, 4, 3, 2]
    num_sp_samples = 1  # 10
    num_cv_splits = 2  # 5
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
    #data = data[:, [4, 5, 6, 13]]

    # Define set of parameters to test.
    train_size = int(len(data) * (num_cv_splits - 1) / num_cv_splits)
    percent_num_supp = [0.02, 0.05, 0.1]
    num_supp_list = [int(train_size * i) for i in percent_num_supp]
    max_iter = 301 #501  # 301
    lr = 0.5 #0.01         # 0.5
    burnin = 5000


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
    #data = data[:, [4, 6, 7, 10]]

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
    #data = data[:, [1, 6, 7, 8]]

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


if __name__ == '__main__':
    main()
