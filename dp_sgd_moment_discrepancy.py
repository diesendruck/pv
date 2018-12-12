""" This script runs differential privacy SGD (moment accountant) on a GAN
    with moment discrepancy loss.
"""

import argparse
from time import time
import os
import pdb
import shutil
import sys
import numpy as np
from numpy.linalg import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist
from scipy.stats import truncnorm

import tensorflow as tf
layers = tf.layers


sys.path.append('/home/maurice/mmd')
from mmd_utils import (compute_mmd, compute_kmmd, compute_cmd,
                       compute_joint_moment_discrepancy,
                       compute_noncentral_moment_discrepancy,
                       compute_moments, compute_central_moments,
                       compute_kth_central_moment, MMD_vs_Normal_by_filter,
                       dp_sensitivity_to_expectation)

from dp_optimizer import DPGradientDescentOptimizer
from sanitizer import AmortizedGaussianSanitizer, ClipOption
from accountant import GaussianMomentsAccountant, EpsDelta
from utils import NetworkParameters, LayerParameters, BuildNetwork


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='cmd_gan',
                    choices=['cmd_gan'])
parser.add_argument('--cmd_variation', type=str, default=None,
                    choices=['dp_sgd', 'onetime_noisy', 'onetime_noisy_joint'])
parser.add_argument('--default_gradient_l2norm_bound', type=float, default=4.0)
parser.add_argument('--laplace_eps', type=float, default=2.0)
parser.add_argument('--sgd_eps', type=float, default=1.0)
parser.add_argument('--sgd_delta', type=float, default=1e-5)
parser.add_argument('--sgd_sigma', type=float, default=4.0)
parser.add_argument('--data_num', type=int, default=10000)
parser.add_argument('--data_dim', type=int, default=1)
parser.add_argument('--percent_train', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--gen_num', type=int, default=200)
parser.add_argument('--width', type=int, default=20,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=10,
                    help='num of generator layers')
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--log_step', type=int, default=1000)
parser.add_argument('--max_step', type=int, default=200000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_update_step', type=int, default=10000)
parser.add_argument('--lr_minimum', type=float, default=1e-6)
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop',
                             'adadelta'])
parser.add_argument('--data_file', type=str, default='')
parser.add_argument('--k_moments', type=int, default=2)
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--load_existing', default=False, action='store_true',
                    dest='load_existing')
args = parser.parse_args()
model_type = args.model_type
cmd_variation = args.cmd_variation
default_gradient_l2norm_bound = args.default_gradient_l2norm_bound
laplace_eps = args.laplace_eps
sgd_eps = args.sgd_eps
sgd_delta = args.sgd_delta
sgd_sigma = args.sgd_sigma
data_num = args.data_num
data_dim = args.data_dim
percent_train = args.percent_train
batch_size = args.batch_size
gen_num = args.gen_num
width = args.width
depth = args.depth
z_dim = args.z_dim
log_step = args.log_step
max_step = args.max_step
learning_rate = args.learning_rate
lr_update_step = args.lr_update_step
lr_minimum = args.lr_minimum
optimizer = args.optimizer
data_file = args.data_file
k_moments = args.k_moments
sigma = args.sigma
tag = args.tag
load_existing = args.load_existing

activation = tf.nn.elu
data_num = int(percent_train * data_num)  # Update based on % train.


def mm(arr):
    """Prints min and max of array."""
    print '  {}, {}'.format(np.min(arr), np.max(arr))


def get_random_z(gen_num, z_dim, for_training=True):
    """Generates 2d array of noise input data."""
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    if for_training:
        return np.random.normal(size=[gen_num, z_dim])
    else:
        return truncnorm.rvs(-3, 3, size=[gen_num, z_dim])


def dense(x, width, activation, batch_residual=False, use_bias=False, name=None,
          bounds=None):
    """Wrapper on fully connected TensorFlow layer.
    
    Args:
      x: Layer input.
      width: Width of output layer.
      activation: TensorFlow activation function.
      batch_residual: Flag to use batch residual output.
      use_bias: Flag to use bias in fully connected layer.
      bounds: List of min and max scalar values to clip to. [min, max]
    """
    # TODO: Should any of this use bias?
    if batch_residual:
        option = 1
        if option == 1:
            x_ = layers.dense(x, width, activation=activation, use_bias=use_bias)
            if bounds is not None:
                x_ = tf.clip_by_value(x_, bounds[0], bounds[1])
            r = layers.batch_normalization(x_) + x
        elif option == 2:
            x_newdim = layers.dense(x, width, activation=activation,
                                    use_bias=use_bias, name=name)
            x_newdim = layers.batch_normalization(x_newdim)
            x_newdim = tf.nn.relu(x_newdim)
            x = layers.dense(x_newdim, width, activation=activation,
                              use_bias=use_bias, name=name)
            x = layers.batch_normalization(x)
            x = tf.nn.relu(x)
            r = x + x_newdim
    else:
        x_ = layers.dense(x, width, activation=activation, use_bias=use_bias)
        if bounds is not None:
            x_ = tf.clip_by_value(x_, bounds[0], bounds[1])
        r = layers.batch_normalization(x_)
    return r


def autoencoder(x, width=3, depth=3, activation=tf.nn.elu, z_dim=3,
                reuse=False, normed_weights=False, normed_encs=False,
                bounds=None):
    """Autoencodes input via a bottleneck layer h."""
    out_dim = x.shape[1]
    with tf.variable_scope('encoder', reuse=reuse) as vs_enc:
        x = dense(x, width, activation=activation, bounds=bounds)

        for idx in range(depth - 1):
            # TODO: Should this use batch resid, and is it defined properly?
            x = dense(x, width, activation=activation, batch_residual=True,
                      bounds=bounds)

        h = dense(x, z_dim, activation=None, bounds=bounds)

    with tf.variable_scope('decoder', reuse=reuse) as vs_dec:

        x = dense(h, width, activation=activation, name='hidden',
                  bounds=bounds)

        for idx in range(depth - 1):
            x = dense(x, width, activation=activation, batch_residual=True,
                      bounds=bounds)

        ae = dense(x, out_dim, activation=None, bounds=bounds)

    vars_enc = tf.contrib.framework.get_variables(vs_enc)
    vars_dec = tf.contrib.framework.get_variables(vs_dec)

    return h, ae, vars_enc, vars_dec


def generator(z_in, width=3, depth=3, activation=tf.nn.elu, out_dim=2,
              reuse=False, bounds=None):
    """Decodes. Generates output, given noise input."""
    bounds = None 
    if bounds == None:
        print 'No bounds on Generator'
    with tf.variable_scope('generator', reuse=reuse) as vs_g:
        x = dense(z_in, width, activation=activation, bounds=bounds,
                  batch_residual=False, use_bias=False)

        for idx in range(depth - 1):
            # TODO: Should this use batch resid, and is it defined properly?
            x = dense(x, width, activation=activation, bounds=bounds,
                      batch_residual=False, use_bias=False)

        out = dense(x, out_dim, activation=None, bounds=bounds,
                    batch_residual=False, use_bias=False)
    vars_g = tf.contrib.framework.get_variables(vs_g)
    return out, vars_g


def fully_connected(x, num_units, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable(
            "weights",
            [x.shape[1], num_units],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(num_units)))
        b = tf.get_variable(
            "biases",
            [num_units],
            initializer=tf.constant_initializer(0.0))
        return tf.nn.relu(tf.matmul(x, w) + b)

def generator_v2(inputs, network_parameters):
    #with tf.variable_scope('generator', reuse=reuse) as vs_g:
    #with tf.variable_scope('generator_v2', reuse=tf.AUTO_REUSE):
        #out, _, training_params = BuildNetwork(inputs, network_parameters)

    num_inputs = network_parameters.input_size
    outputs = inputs
    for layer_parameters in network_parameters.layer_parameters:
        # Get number of hidden units in this layer.
        num_units = layer_parameters.num_units

        # Do the layer.
        outputs = fully_connected(outputs, num_units, layer_parameters.name)

        # Set output dim of this layer as input dim for next layer.
        num_inputs = num_units

    return outputs 


def load_checkpoint(saver, sess, checkpoint_dir):
    """Restores weights from pre-trained model."""
    import re
    print ' [*] Reading checkpoints...'
    print '     {}'.format(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        #counter = int(''.join([i for i in ckpt_name if i.isdigit()]))
        counter = int(ckpt_name.split('-')[-1])
        print ' [*] Success to read {}'.format(ckpt_name)
        return True, counter
    else:
        print ' [*] Failed to find a checkpoint'
        return False, 0


def load_normed_data(data_num, percent_train, data_file=None, save=False):
    """Generates data, and returns it normalized, along with helper objects."""
    # Load data.
    if data_file and not save:
        if data_file.endswith('npy'):
            data_raw = np.load(data_file)
            data_raw = data_raw[:10000]
            print 'Using first 10000 from {}.'.format(data_file)
        elif data_file.endswith('txt'):
            data_raw = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
            #data_raw = np.random.permutation(data_raw)
    else:
        if data_dim == 1:
            data_raw = np.zeros((data_num, 1))
            for i in range(data_num):
                # Pick a Gaussian, then generate from that Gaussian.
                cluster_i = np.random.binomial(1, 0.4)  # NOTE: Setting p=0/p=1 chooses one cluster.
                if cluster_i == 0:
                    data_raw[i] = np.random.normal(0, 2)
                    #data_raw[i] = np.random.gamma(7, 2)
                else:
                    data_raw[i] = np.random.normal(6, 2)
        elif data_dim == 2:
            #design = 'noisy_sin'
            #design = 'two_gaussians'
            design = 'two_gaussians'
            print('Using data set {}'.format(design))

            if design == 'two_gaussians':
                def sample_c1():
                    sample = np.random.multivariate_normal(
                        [0., 0.], [[1., 0.], [0., 1.]], 1)
                    return sample
                def sample_c2():
                    sample = np.random.multivariate_normal(
                        [2., 2.], [[1, 0.2], [0.2, 1]], 1)
                    return sample
                data_raw = np.zeros((data_num, 2))
                for i in range(data_num):
                    if np.random.binomial(1, 0.5):  # 2nd param controls mixture.
                        data_raw[i] = sample_c1()
                    else:
                        data_raw[i] = sample_c2()

            elif design == 'noisy_sin':
                x = np.linspace(0, 10, data_num)
                y = np.sin(x) + np.random.normal(0, 0.5, len(x))
                data_raw = np.hstack((np.expand_dims(x, axis=1),
                                      np.expand_dims(y, axis=1)))
                data_raw = data_raw[np.random.permutation(data_num)]
            elif design == 'uniform':
                data_raw = np.zeros((data_num, 2))
                for i in range(data_num):
                    if np.random.binomial(1, 0.5):
                        x_sample = np.random.uniform(0, 1)
                        y_sample = np.random.uniform(0, 1)
                    else:
                        x_sample = np.random.uniform(-1, 0)
                        y_sample = np.random.uniform(-1, 0)
                    data_raw[i] = [x_sample, y_sample]
        if save == True:
            np.save('data_raw_2d.npy', data_raw)

    # Do data normalization.
    data_raw_mean = np.mean(data_raw, axis=0)
    data_raw_std = np.std(data_raw, axis=0)
    data_normed = (data_raw - data_raw_mean) / data_raw_std

    # Split data into train and test.
    num_train = int(percent_train * data_normed.shape[0])
    data = data_normed[:num_train]
    data_test = data_normed[num_train:]

    # Set up a few helpful constants.
    data_num = data.shape[0]
    data_test_num = data_test.shape[0]
    out_dim = data.shape[1]

    return (data, data_test, data_num, data_test_num, out_dim,
            data_raw_mean, data_raw_std)


def print_baseline_moment_stats(k_moments, data_num, percent_train):
    """Compute baseline statistics on moments for data set."""
    d, _, d_num, _, _, d_raw_mean, d_raw_std = load_normed_data(
        data_num, percent_train)
    d = d * d_raw_std + d_raw_mean
    baseline_moments = compute_moments(d, k_moments)
    for j in range(k_moments):
        print 'Moment {}: {}'.format(j+1, baseline_moments[j])


def make_fixed_batches_and_sensitivities(data, batch_size, k_moments):
    # Partition data into fixed batches.
    data_dim = data.shape[1]
    fixed_batches = np.array(
        [data[i:i + batch_size] for i in xrange(0, len(data), batch_size)])
    fixed_batches = np.array(
        [b for b in fixed_batches if len(b) == batch_size])
    num_batches = len(fixed_batches)

    # Compute k moment sensitivities for entire data set.
    # Then tile the max sensitivity so it applies to all fixed batches.
    moment_sensitivities = []
    for k in range(1, k_moments + 1):
        mk_sens = np.max(np.power(np.abs(data), k), axis=0) / batch_size
        moment_sensitivities.append(mk_sens)
    m_sens = np.expand_dims(moment_sensitivities, axis=0)
    fixed_batches_moment_sensitivities = np.tile(m_sens, [num_batches, 1, 1])

    # Compute k joint-moment sensitivities for entire data set.
    # Then copy the max sensitivity so it applies to all fixed batches.
    jmoment_sensitivities = []
    for k in range(1, k_moments + 1):
        jmk_sens = (np.max(np.power(np.abs(np.prod(data, axis=1)), k), axis=0) /
            batch_size)
        jmoment_sensitivities.append(jmk_sens)
    jm_sens = np.reshape(jmoment_sensitivities, [1, -1, 1])
    fixed_batches_jmoment_sensitivities = np.tile(jm_sens, [num_batches, 1, 1])

    # Compute quantile sensitivities for the entire data set, i.e. the
    # the difference between the quantile value and the next value in a sorted
    # array. In the end, tile the sensitivity so it applies to all batches.
    n = data_num
    quantiles = [0.2, 0.5, 0.8]
    global_quantile_values = np.zeros((data_dim, len(quantiles)))
    quantile_sensitivities = np.zeros((data_dim, len(quantiles)))
    for dim in range(data_dim):
        component_data = sorted(data[:, dim])
        for q_i, q_v in enumerate(quantiles):
            index_q = int(np.ceil(n * q_v))
            global_quantile_values[dim, q_i] = component_data[index_q] 
            quantile_sensitivities[dim, q_i] = (
                component_data[index_q + 1] - component_data[index_q])

    return (fixed_batches,
            fixed_batches_moment_sensitivities,
            fixed_batches_jmoment_sensitivities,
            global_quantile_values,
            quantile_sensitivities)


def make_fbo_noisy_moments(fixed_batches, fixed_batches_moment_sensitivities,
                           k_moments, laplace_eps, allocation=None):
    """Sets up true and noisy moments per batch.

    Args:
      fixed_batches: Array of data batches, [batch_num, batch_size, data_dim].
      fixed_batches_moment_sensitivities: Array of sensitivities of moments,
        [batch_num, k_moments, data_dim].
      k_moments: Integer number of moments to compute.
      laplace_eps: Float, overall privacy budget.
      allocation: List of values, to be normalized, that allocate the privacy
        budget among moments.

    Returns:
      fixed_batches_onetime_noisy_moments: Array of fixed batch noisy moments,
        according to sensitivities of each moment, [batch_num, k_moments].
    """
    data_dim = fixed_batches.shape[-1]
    fixed_batches_moments = np.zeros(
        (len(fixed_batches), k_moments, data_dim), dtype=np.float32)
    fixed_batches_onetime_noisy_moments = np.zeros(
        (len(fixed_batches), k_moments, data_dim), dtype=np.float32)

    # Choose allocation of budget.
    if allocation is None:
        allocation = [1] * k_moments
    print('Privacy budget and allocation: {}, {}\n'.format(
        laplace_eps, allocation))
    eps_ = laplace_eps * (np.array(allocation) / float(np.sum(allocation)))
    assert len(eps_) == k_moments, 'allocation length must match moment num'

    # Get moments and noisy version for each batch.
    for batch_num, batch in enumerate(fixed_batches):
        batch_sensitivities = fixed_batches_moment_sensitivities[batch_num]
        # Each moment within that batch.
        raw_moments = np.zeros((k_moments, data_dim))
        noisy_moments = np.zeros((k_moments, data_dim))
        for k in range(1, k_moments + 1):
            mk_sens = batch_sensitivities[k - 1]  
            # Sample laplace noise for each dimension of data -- scale
            # param takes vector of laplace scales and outputs
            # corresponding values.
            mk_laplace = np.random.laplace(loc=0, scale=mk_sens/eps_[k-1])
            mk = np.mean(np.power(batch, k), axis=0)
            mk_noisy = mk + mk_laplace
            raw_moments[k - 1] = mk
            noisy_moments[k - 1] = mk_noisy
        fixed_batches_moments[batch_num] = raw_moments
        fixed_batches_onetime_noisy_moments[batch_num] = noisy_moments
    print ' Sample: RAW moments'
    print fixed_batches_moments[0]
    print ' Sample: NOISY moments'
    print fixed_batches_onetime_noisy_moments[0]
    return fixed_batches_onetime_noisy_moments


def make_fbo_noisy_jmoments(fixed_batches, fixed_batches_jmoment_sensitivities,
                            k_moments, laplace_eps, allocation=None):
    """Sets up true and noisy joint moments per batch.

    Args:
      fixed_batches: Array of data batches, [batch_num, batch_size, data_dim].
      fixed_batches_jmoment_sensitivities: Array of sensitivities of joint
        moments, [batch_num, k_moments].
      k_moments: Integer number of moments to compute.
      laplace_eps: Float, overall privacy budget.
      allocation: List of values, to be normalized, that allocate the privacy
        budget among moments.

    Returns:
      fixed_batches_onetime_noisy_jmoments: Array of fixed batch noisy joint
        moments, according to sensitivities of each joint moment,
        [batch_num, k_moments].
    """
    data_dim = fixed_batches.shape[-1]
    fixed_batches_jmoments = np.zeros(
        (len(fixed_batches), k_moments), dtype=np.float32)
    fixed_batches_onetime_noisy_jmoments = np.zeros(
        (len(fixed_batches), k_moments), dtype=np.float32)

    # Choose allocation of budget.
    if allocation is None:
        allocation = [1] * k_moments
    print('Privacy budget and allocation: {}, {}\n'.format(
        laplace_eps, allocation))
    eps_ = laplace_eps * (np.array(allocation) / float(np.sum(allocation)))
    assert len(eps_) == k_moments, 'allocation length must match moment num'

    # Get moments and noisy version for each batch.
    for batch_num, batch in enumerate(fixed_batches):
        batch_sensitivities = fixed_batches_jmoment_sensitivities[batch_num]
        # Each moment within that batch.
        raw_jmoments = np.zeros(k_moments)
        noisy_jmoments = np.zeros(k_moments)
        for k in range(1, k_moments + 1):
            mk_sens = batch_sensitivities[k - 1]  
            # Sample laplace noise for each dimension of data -- scale
            # param takes vector of laplace scales and outputs
            # corresponding values.
            mk_laplace = np.random.laplace(loc=0, scale=mk_sens/eps_[k-1])
            mk = np.mean(np.power(np.prod(batch, axis=1), k), axis=0)
            mk_noisy = mk + mk_laplace
            raw_jmoments[k - 1] = mk
            noisy_jmoments[k - 1] = mk_noisy
        fixed_batches_jmoments[batch_num] = raw_jmoments
        fixed_batches_onetime_noisy_jmoments[batch_num] = noisy_jmoments
    print ' Sample: RAW jmoments'
    print fixed_batches_jmoments[0]
    print ' Sample: NOISY jmoments'
    print fixed_batches_onetime_noisy_jmoments[0]
    return fixed_batches_onetime_noisy_jmoments


# TODO: Complete this for quantiles.
def make_noisy_quantiles(global_quantile_values, 
                         quantile_sensitivities,
                         laplace_eps):
    """Adds Laplace noise to true, global quantile values.

    Args:
      global_quantile_values: Array of true quantile values,
        [data_dim, num_quantiles].
      quantile_sensitivities: Array of sensitivities of quantiles,
        [data_dim, num_quantiles].
      laplace_eps: Float, overall privacy budget.

    Returns:
      onetime_noisy_quantiles: Array of noisy quantiles, according to
        sensitivities of each quantile, [batch_num, data_dim, num_quantiles].
    """
    # Simple case: Just perturb global quantiles with Laplace noise.
    num_quantiles = quantile_sensitivities.shape[-1]
    onetime_noisy_quantiles = np.zeros(global_quantile_values.shape,
                                       dtype=np.float32)

    # Sample laplace noise for each dimension of data -- scale
    # param takes vector of laplace scales and outputs
    # corresponding values.
    scale_params = quantile_sensitivities / laplace_eps
    laplace_noise = np.random.laplace(loc=0, scale=scale_params)
    onetime_noisy_quantiles = global_quantile_values + laplace_noise

    print ' RAW quantiles'
    print global_quantile_values
    print ' NOISY quantiles'
    print onetime_noisy_quantiles
    return onetime_noisy_quantiles


def get_noisy_mean_cov(fixed_batches, laplace_eps):
    """Computes noisy mean and covariance for fixed batches, based on
    sensitivities of each across all batches.
    
    Args:
      fixed_batches: NumPy array of fixed batches of inputs, of dimension
        [num_batches, batch_size, input_dim].
      laplace_eps: Float, differential privacy epsilon.

    Returns:
      noisy_means: NumPy array of noisy means.
      noisy_covs: NumPy array of flattened noisy covs.
    """
    num_batches = fixed_batches.shape[0]
    batch_size = fixed_batches.shape[1]
    input_dim = fixed_batches.shape[2]

    means = np.zeros((num_batches, input_dim), dtype=np.float32)
    covs = np.zeros((num_batches, input_dim, input_dim), dtype=np.float32)
    noisy_means = np.zeros((num_batches, input_dim), dtype=np.float32)
    noisy_covs = np.zeros((num_batches, input_dim, input_dim), dtype=np.float32)

    # Store each batch's mean and vectorized covariance.
    for i, b in enumerate(fixed_batches):
        means[i] = np.mean(b, axis=0)
        covs[i] = np.cov(b, rowvar=False)

    # Compute sensitivities of means and covs.
    # For the whole set, compute sensitivity of each moment.
    mean_sensitivity = np.max(np.abs(means), axis=0) / batch_size
    cov_sensitivity = (
        np.max(
            np.abs(
                np.reshape(covs, [num_batches, -1])),  # Vectorized covariances.
            axis=0) /
        batch_size)

    # Compute noisy moments for each batch.
    for i, b in enumerate(fixed_batches):
        natural_mean = means[i]
        laplace_noise_mean = \
            np.random.laplace(loc=0, scale=mean_sensitivity/laplace_eps)
        noisy_means[i] = natural_mean + laplace_noise_mean
        print(natural_mean)
        print(natural_mean + laplace_noise_mean)

        natural_cov = covs[i]
        valid_cov = False
        tries = 0
        tries_limit = 5
        while not valid_cov and tries < tries_limit:
            # Sample and laplace noise to cov. Make cov symmetric. Verify PSD.
            laplace_noise_cov = \
                np.random.laplace(loc=0, scale=cov_sensitivity/laplace_eps)
            noisy_cov_vec = np.reshape(natural_cov, [1, -1]) + laplace_noise_cov
            noisy_cov = np.reshape(noisy_cov_vec, [input_dim, input_dim])
            indices_lower = np.tril_indices(input_dim, -1)
            noisy_cov[indices_lower] = noisy_cov.T[indices_lower]
            print(natural_cov)
            print(noisy_cov)
            if np.all(np.linalg.eigvals(noisy_cov) > 0):
                valid_cov = True
            else:
                tries += 1
                print('----NOT PSD {}----'.format(tries))
        noisy_covs[i] = noisy_cov

    # Check that outputs are somewhat close for large eps.
    print ' Sample: natural and noisy means'
    print means[:3], noisy_means[:3]
    print ' Sample: natural and noisy covs'
    print covs[:3], noisy_covs[:3]

    return noisy_means, noisy_covs


def unnormalize(data_normed, data_raw_mean, data_raw_std):
    """Unnormalizes data based on mean and std."""
    return data_normed * data_raw_std + data_raw_mean


def prepare_dirs(load_existing):
    """Creates directories for logs, checkpoints, and plots."""
    log_dir = 'logs/logs_{}'.format(tag)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    plot_dir = os.path.join(log_dir, 'plots')
    g_out_dir = os.path.join(log_dir, 'g_out')
    if os.path.exists(log_dir) and not load_existing:
        shutil.rmtree(log_dir)
    for path in [log_dir, checkpoint_dir, plot_dir, g_out_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, checkpoint_dir, plot_dir, g_out_dir


def prepare_logging(log_dir, checkpoint_dir, sess):
    """Sets up TensorFlow logging."""
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'),
                                           sess.graph)
    step = tf.Variable(0, name='step', trainable=False)
    sv = tf.train.Supervisor(logdir=checkpoint_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             summary_writer=summary_writer,
                             save_model_secs=300,
                             global_step=step,
                             ready_for_local_init_op=None)
    return saver, summary_writer


def add_nongraph_summary_items(summary_writer, step, dict_to_add):
    """Adds to list of summary items during logging."""
    for k, v in dict_to_add.iteritems():
        summ = tf.Summary()
        summ.value.add(tag=k, simple_value=v)
        summary_writer.add_summary(summ, step)
    summary_writer.flush()


def avg_nearest_neighbor_distance(candidates, references, flag='noflag'):
    """Measures distance from candidate set to a reference set.
    NOTE: This is not symmetric!

    For each element in candidate set, find distance to nearest neighbor in
    reference set. Return the average of these distances.

    Args:
      candidates: Numpy array of candidate points. (num_points x point_dim)
      references: Numpy array of reference points. (num_points x point_dim)

    Returns:
      avg_dist: Float, average over distances.
      distances: List of distances to nearest neighbor for each candidate.
    """
    distances = []
    for i in xrange(candidates.shape[0]):
        c_i = tf.gather(candidates, [i])
        distances_from_i = tf.norm(c_i - references, axis=1)
        d_from_i_reshaped = tf.reshape(distances_from_i, [1, -1])  # NEW

        assert d_from_i_reshaped.shape.as_list() == [1, references.shape[0]]
        distances_negative = -1.0 * d_from_i_reshaped
        #distances_negative = -1.0 * distances_from_i  # OLD
        smallest_dist_negative, _ = tf.nn.top_k(distances_negative, name=flag)
        assert smallest_dist_negative.shape.as_list() == [1, 1]
        smallest_dist = -1.0 * smallest_dist_negative[0]

        distances.append(smallest_dist)

    avg_dist = tf.reduce_mean(distances)
    return avg_dist, distances


def evaluate_presence_risk(train, test, sim, ball_radius=1e-2):
    """Assess privacy of simulations.

    Compute True Pos., True Neg., False Pos., and False Neg. rates of
    finding a neighbor in the simulations, for each of a subset of training
    data and a subset of test data.

    Args:
      train: Numpy array of all training data.
      test: Numpy array of all test data (smaller than train).
      sim: Numpy array of simulations.
      ball_radius: Float, distance from point to sim that qualifies as match.

    Return:
      sensitivity: Float of TP / (TP + FN).
      precision: Float of TP / (TP + FP).
    """
    # TODO: Sensitivity as a loss, rather than just be reported?
    # TODO: Count nearest neighbors, rather than presence in epsilon-ball?
    assert len(test) < len(train), 'test should be smaller than train'
    num_samples = len(test)
    compromised_records = train[:num_samples]
    tp, tn, fp, fn = 0, 0, 0, 0

    # Count true positives and false negatives.
    for i in compromised_records:
        distances_from_i = norm(i - sim, axis=1)
        has_neighbor = np.any(distances_from_i < ball_radius)
        if has_neighbor:
            tp += 1
        else:
            fn += 1
    # Count false positives and true negatives.
    for i in test:
        distances_from_i = norm(i - sim, axis=1)
        has_neighbor = np.any(distances_from_i < ball_radius)
        if has_neighbor:
            fp += 1
        else:
            tn += 1

    sensitivity = float(tp) / (tp + fn)
    precision = float(tp + 1e-10) / (tp + fp + 1e-10)
    false_positive_rate = float(fp) / (fp + tn)
    return (sensitivity, precision, false_positive_rate, tp, fn, fp, tn)


def load_network_parameters(z_dim, default_gradient_l2norm_bound,
                            depth, width, out_dim):
    network_parameters = NetworkParameters()
    network_parameters.input_size = z_dim
    network_parameters.default_gradient_l2norm_bound = \
        default_gradient_l2norm_bound
    for i in xrange(depth):
        hidden = LayerParameters()
        hidden.name = "hidden%d" % i
        hidden.num_units = width
        hidden.relu = True
        hidden.with_bias = False
        hidden.trainable = True
        network_parameters.layer_parameters.append(hidden)

    gen = LayerParameters()
    gen.name = 'gen'
    gen.num_units = out_dim 
    gen.relu = False
    gen.with_bias = False
    network_parameters.layer_parameters.append(gen)
    return network_parameters


def build_model_cmd_gan(batch_size, gen_num, data_num, data_test_num, out_dim,
                        z_dim, cmd_span_const,
                        fixed_batches_moment_sensitivities,
                        quantile_sensitivities,
                        bounds):
    """Builds model for Central Moment Discrepancy as adversary."""

    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(
        tf.float32, [data_test_num, out_dim], name='x_precompute')

    x_test_precompute = tf.placeholder(
        tf.float32, [data_test_num, out_dim], name='x_test_precompute')

    avg_dist_x_to_x_test_precomputed, distances_xt_xp = \
        avg_nearest_neighbor_distance(x_precompute, x_test_precompute)


    # Placeholders for regular training.
    x = tf.placeholder(tf.float32, [batch_size, out_dim], name='x')
    z = tf.placeholder(tf.float32, [gen_num, z_dim], name='z')
    z_readonly = tf.placeholder(tf.float32, [data_num, z_dim], name='z_readonly')
    x_test = tf.placeholder(tf.float32, [batch_size, out_dim], name='x_test')

    avg_dist_x_to_x_test = tf.placeholder(
        tf.float32, shape=(), name='avg_dist_x_to_x_test')

    prog_cmd_coefs = tf.placeholder(
        tf.float32, shape=(k_moments), name='prog_cmd_coefs')

    mmd_to_cmd_indicator = tf.placeholder(
        tf.float32, shape=(), name='mmd_to_cmd_indicator')


    # Node to update learning rate.
    lr = tf.Variable(learning_rate, name='lr', trainable=False)
    lr_update = tf.assign(lr, tf.maximum(lr * 0.8, lr_minimum),
                          name='lr_update')


    # Get generator output.
    use_generator_v2 = 1
    if not use_generator_v2:
        g, g_vars = generator(
            z, width=width, depth=depth, activation=activation, out_dim=out_dim,
            bounds=bounds)
        g_readonly, _ = generator(
            z_readonly, width=width, depth=depth, activation=activation,
            out_dim=out_dim, reuse=True, bounds=bounds)
    else:
        # Dense network using Google Research code from Github.
        # https://github.com/tensorflow/models/blob/master/research/
        #   differential_privacy/dp_sgd/dp_optimizer/utils.py
        network_parameters = load_network_parameters(
            z_dim, default_gradient_l2norm_bound, depth, width, out_dim)
        with tf.variable_scope('generator_v2') as scope:
            g = generator_v2(z, network_parameters)
            g_readonly = generator_v2(z_readonly, network_parameters)


    #############################
    # Begin: Define the CMD loss.
    cmd_k, cmd_k_terms = compute_cmd(
        x, g, k_moments=k_moments, use_tf=True, cmd_span_const=cmd_span_const,
        return_terms=True)
    cmd_k_minus_k_plus_1 = compute_cmd(
        x, g, k_moments=k_moments+1, use_tf=True, cmd_span_const=cmd_span_const)
    mmd = compute_mmd(x, g, use_tf=True, slim_output=True)
    nmd_k = compute_noncentral_moment_discrepancy(
        x, g, k_moments=k_moments, use_tf=True, cmd_span_const=cmd_span_const)
    # NoncentralMD on one-time-noised empirical data moments.
    if cmd_variation == 'onetime_noisy':
        batch_id = tf.placeholder(tf.int32, shape=(), name='batch_id')
        fbo_noisy_moments = tf.placeholder(
            tf.float32, [None, k_moments, out_dim], name='fbo_noisy_moments')
        cmd_adjusted = compute_noncentral_moment_discrepancy(
            x, g, k_moments=k_moments, use_tf=True,
            cmd_span_const=cmd_span_const, batch_id=batch_id,
            fbo_noisy_moments=fbo_noisy_moments)
    # Joint MD on one-time-noised empirical data moments.
    elif cmd_variation == 'onetime_noisy_joint':
        batch_id = tf.placeholder(tf.int32, shape=(), name='batch_id')
        fbo_noisy_moments = tf.placeholder(
            tf.float32, [None, k_moments, out_dim], name='fbo_noisy_moments')
        fbo_noisy_jmoments = tf.placeholder(
            tf.float32, [None, k_moments, 1], name='fbo_noisy_jmoments')
        num_quantiles = quantile_sensitivities.shape[-1]
        noisy_quantiles = tf.placeholder(
            tf.float32, [out_dim, num_quantiles], name='noisy_quantiles')
        # Noncentral moment discrepancy.
        d1 = compute_noncentral_moment_discrepancy(
            x, g, k_moments=k_moments, use_tf=True,
            cmd_span_const=cmd_span_const, batch_id=batch_id,
            fbo_noisy_moments=fbo_noisy_moments)
        # Joint moment discrepancy.
        d2 = compute_joint_moment_discrepancy(
            x, g, k_moments=k_moments, use_tf=True,
            cmd_span_const=cmd_span_const, batch_id=batch_id,
            fbo_noisy_moments=fbo_noisy_moments,
            fbo_noisy_jmoments=fbo_noisy_jmoments,
            noisy_quantiles=noisy_quantiles)
        # Combine the noncentral and joint discrepancies.
        cmd_adjusted = 1 * d1 + .1 * d2
    # Normal CMD loss.
    else:
        cmd_adjusted = cmd_k
        d1 = 0.
        d2 = 0.
        batch_id = None
        fbo_noisy_moments = None
        fbo_noisy_jmoments = None
        noisy_quantiles = None
    # End: Define the CMD loss.
    #############################


    # Define optimization operations.
    if cmd_variation == 'dp_sgd':
        ####################################
        # Begin: Differentially private SGD.
        # In classification example on Github, authors take average of cross-
        # entropy loss across batch, claiming the "actual cost is the average
        # across the examples". Here, the CMD is already an expectation over
        # the samples, so it's unclear whether it should be scaled.
        cmd = cmd_k
        g_loss = nmd_k  # NOTE: Using Noncentral Moment Discrepancy.

        # Define effective training size, given fixed batches.
        num_training = batch_size * len(fixed_batches_moment_sensitivities)

        # Instantiate the accountant.
        priv_accountant = GaussianMomentsAccountant(num_training)

        # Define sigma.
        sigma = sgd_sigma
        
        # Instantiate the sanitizer.
        gaussian_sanitizer = AmortizedGaussianSanitizer(
            priv_accountant,
            [default_gradient_l2norm_bound / batch_size, True])  # / batch_size?

        # Setting clip options for each var. For now, this does nothing, as all
        # vars take default option.
        #for var in training_params:
        #    if "gradient_l2norm_bound" in training_params[var]:
        #        l2bound = training_params[var]["gradient_l2norm_bound"] / batch_size
        #        gaussian_sanitizer.set_option(var, sanitizer.ClipOption(l2bound,
        #                                                                True))

        # Constants for optimization step.
        #lr = tf.placeholder(tf.float32)
        eps = tf.placeholder(tf.float32)
        delta = tf.placeholder(tf.float32)

        # Define optimization node.
        g_optim = DPGradientDescentOptimizer(
            lr,
            [eps, delta],
            gaussian_sanitizer,
            sigma=sigma,
            batches_per_lot=1).minimize(g_loss)

        # End: Differentially private SGD.
        ####################################
    else:
        cmd = cmd_k
        g_loss = nmd_k  # NOTE: Using Noncentral Moment Discrepancy.

        if optimizer == 'adagrad':
            g_opt = tf.train.AdagradOptimizer(lr)
        elif optimizer == 'adam':
            g_opt = tf.train.AdamOptimizer(lr)
        elif optimizer == 'rmsprop':
            g_opt = tf.train.RMSPropOptimizer(lr)
        elif optimizer == 'adadelta':
            g_opt = tf.train.AdadeltaOptimizer(lr)
        else:
            g_opt = tf.train.GradientDescentOptimizer(lr)

        # Define optim nodes.
        # TODO: TEST CLIPPED cmd_gan_generator GENERATOR.
        clip = 1
        if clip:
            g_grads_, g_vars_ = zip(*g_opt.compute_gradients(g_loss, var_list=g_vars))
            g_grads_clipped_ = tuple(
                [tf.clip_by_value(grad, -0.001, 0.001) for grad in g_grads_])
            g_optim = g_opt.apply_gradients(zip(g_grads_clipped_, g_vars_))
        else:
            g_optim = g_opt.minimize(g_loss, var_list=g_vars)


    # Get diagnostics. Compare distances between data, heldouts, and gen.
    avg_dist_g_to_x, distances_g_x = avg_nearest_neighbor_distance(g, x)
    avg_dist_x_to_g, distances_x_g = avg_nearest_neighbor_distance(x, g)
    avg_dist_x_test_to_g, distances_xt_g = avg_nearest_neighbor_distance(x_test, g)
    loss1 = avg_dist_x_to_x_test - avg_dist_x_to_g
    loss2 = avg_dist_x_test_to_g - avg_dist_x_to_g

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/g_loss", g_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/cmd", cmd),
	tf.summary.scalar("loss/d1", d1),
	tf.summary.scalar("loss/d2", d2),
	tf.summary.scalar("misc/lr", lr),
    ])

    return (x, z, z_readonly, x_test, x_precompute, x_test_precompute,
            avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
            distances_xt_xp, prog_cmd_coefs, mmd_to_cmd_indicator, cmd_k_terms,
            g, g_readonly, mmd, cmd, loss1, loss2, lr_update, lr, g_optim,
            summary_op, batch_id, fbo_noisy_moments, fbo_noisy_jmoments,
            noisy_quantiles, eps, delta)


def main():
    args = parser.parse_args()
    model_type = args.model_type
    data_num = args.data_num
    percent_train = args.percent_train
    batch_size = args.batch_size
    gen_num = args.gen_num
    width = args.width
    depth = args.depth
    z_dim = args.z_dim
    log_step = args.log_step
    max_step = args.max_step
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    data_file = args.data_file
    tag = args.tag
    load_existing = args.load_existing
    activation = tf.nn.elu

    # Load data and prep dirs.
    (data, data_test, data_num, data_test_num, out_dim, data_raw_mean,
     data_raw_std) = load_normed_data(data_num, percent_train,
                                      data_file=data_file, save=True)
    data_dim = data.shape[1]
    normed_moments_data = compute_moments(data, k_moments=k_moments+1)
    normed_moments_data_test = compute_moments(data_test, k_moments=k_moments+1)
    nmd_zero_indices = np.argwhere(
        norm(np.array(normed_moments_data), axis=1) < 0.1)

    # Compute baseline statistics on moments for data set.
    print_baseline_moment_stats(k_moments, data_num, percent_train)

    # Compute sensitivities for moments, based on fixed batches.
    (fixed_batches,
     fixed_batches_moment_sensitivities,
     fixed_batches_jmoment_sensitivities,
     global_quantile_values,
     quantile_sensitivities) = make_fixed_batches_and_sensitivities(
         data, batch_size, k_moments)
    print('\n\nData size: {}, Batch size: {}, Num batches: {}, '
          'Effective data size: {}\n\n'.format(
               len(data), batch_size, len(fixed_batches),
               batch_size * len(fixed_batches)))

    # ADD ONETIME noise to moment in each batch, according to 
    # fixed_batches_moment_sensitivities.
    allocation = [1] * k_moments
    fixed_batches_onetime_noisy_moments = \
        make_fbo_noisy_moments(
            fixed_batches, fixed_batches_moment_sensitivities, k_moments,
            laplace_eps, allocation=allocation)

    # ADD ONETIME noise to joint moment in each batch, according to 
    # fixed_batches_jmoment_sensitivities.
    allocation = [1] * k_moments
    fixed_batches_onetime_noisy_jmoments = \
        make_fbo_noisy_jmoments(
            fixed_batches, fixed_batches_jmoment_sensitivities, k_moments,
            laplace_eps, allocation=allocation)
    fixed_batches_onetime_noisy_jmoments = np.expand_dims(
        fixed_batches_onetime_noisy_jmoments, axis=2)
    assert (len(fixed_batches_onetime_noisy_moments.shape) ==
            len(fixed_batches_onetime_noisy_jmoments.shape) == 3), (
                'fbo inputs must be 3d tensors')

    # ADD ONETIME noise to quantiles in each batch, according to 
    # quantile_sensitivities.
    # TODO: do this.
    onetime_noisy_quantiles = \
        make_noisy_quantiles(
            global_quantile_values,
            quantile_sensitivities,
            laplace_eps)


    # Get compact interval bounds for CMD computations.
    #cmd_a = np.min(data, axis=0)
    #cmd_b = np.max(data, axis=0)
    #print 'cmd_span_const: {:.2f}'.format(1.0 / (np.abs(cmd_b - cmd_a)))
    data_raw = data * data_raw_std + data_raw_mean
    cmd_a_raw = np.min(data_raw)
    cmd_b_raw = np.max(data_raw)
    cmd_span_const = 1.0 / np.max(pdist(data))
    cmd_a = np.min(data)
    cmd_b = np.max(data)
    print 'cmd_span_const: {:.2f}'.format(cmd_span_const)

    # Prepare logging, checkpoint, and plotting directories.
    log_dir, checkpoint_dir, plot_dir, g_out_dir = prepare_dirs(load_existing)
    save_tag = str(args)
    with open(os.path.join(log_dir, 'save_tag.txt'), 'w') as save_tag_file:
        save_tag_file.write(save_tag)
    print 'Save tag: {}'.format(save_tag)

    # Save data set used for training.
    data_train_unnormed = data * data_raw_std + data_raw_mean
    data_test_unnormed = data_test * data_raw_std + data_raw_mean
    np.save(os.path.join(log_dir, 'data_train.npy'), data_train_unnormed)
    np.save(os.path.join(log_dir, 'data_test.npy'), data_test_unnormed)

    # Save file for outputs in txt form. Also saved later as npy.
    g_out_file = os.path.join(g_out_dir, 'g_out.txt')
    if os.path.isfile(g_out_file):
        os.remove(g_out_file)

    # build_all()
    # Build model.
    (x, z, z_readonly, x_test, x_precompute, x_test_precompute,
     avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed, distances_xt_xp,
     prog_cmd_coefs, mmd_to_cmd_indicator, cmd_k_terms, g, g_readonly, mmd, cmd,
     loss1, loss2, lr_update, lr, g_optim, summary_op, batch_id,
     fbo_noisy_moments, fbo_noisy_jmoments, noisy_quantiles, eps, delta) = \
         build_model_cmd_gan(batch_size, gen_num, data_num, data_test_num,
                             out_dim, z_dim, cmd_span_const,
                             fixed_batches_moment_sensitivities,
                             quantile_sensitivities, bounds=[cmd_a, cmd_b])

    ###########################################################################
    # Start session.
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=sess_config) as sess:
        if cmd_variation == 'dp_sgd':
            # We need to maintain the intialization sequence.
            for v in tf.trainable_variables():
                sess.run(tf.variables_initializer([v]))


        saver, summary_writer = prepare_logging(log_dir, checkpoint_dir, sess)

        sess.run(init_op)

        # load_existing_model().
        if load_existing:
            could_load, checkpoint_counter = load_checkpoint(
                saver, sess, checkpoint_dir)
            if could_load:
                load_step = checkpoint_counter
                print ' [*] Load SUCCESS'
            else:
                print ' [!] Load failed...'
        else:
            load_step = 0

        # Once, compute average distance from heldout data to training data.
        avg_dist_x_to_x_test_precomputed_, _ = sess.run(
            [avg_dist_x_to_x_test_precomputed, distances_xt_xp],
            {x_precompute: data[:len(data_test)],
             x_test_precompute: data_test})

        # Containers to hold empirical and relative errors of moments.
        empirical_moments_gens = np.zeros(
            ((max_step - 0) / log_step, k_moments+1, data_dim))
        relative_error_of_moments = np.zeros(
            ((max_step - 0) / log_step, k_moments+1))
        reom = relative_error_of_moments


        #######################################################################
        # train()
        start_time = time()
        for step in range(load_step, max_step):

            # Set up inputs for all models.
            # OPTION 1: Random batch selection.
            # OPTION 2: Fixed batch selection.
            batch_selection_option = 2

            if batch_selection_option == 1:
                random_batch_data = np.array(
                    [data[d] for d in np.random.choice(len(data), batch_size)])
                _batch_id = None

            elif batch_selection_option == 2:
                _epoch, _batch_id = np.divmod(step, len(fixed_batches))
                if _batch_id == 0 and _epoch % 10 == 0:
                    print ' {}'.format(_epoch)
                random_batch_data = fixed_batches[_batch_id]

            # Fetch test data and z.
            random_batch_data_test = np.array(
                [data_test[d] for d in np.random.choice(
                    len(data_test), batch_size)])
            random_batch_z = get_random_z(gen_num, z_dim)

            # Update shared dict for chosen model.
            if cmd_variation in ['', None]:
                feed_dict = {
                    x: random_batch_data,
                    z: random_batch_z,
                    x_test: random_batch_data_test,
                    avg_dist_x_to_x_test: avg_dist_x_to_x_test_precomputed_,
                    batch_id: _batch_id}
            elif cmd_variation == 'dp_sgd':
                feed_dict = {
                    x: random_batch_data,
                    z: random_batch_z,
                    x_test: random_batch_data_test,
                    avg_dist_x_to_x_test: avg_dist_x_to_x_test_precomputed_,
                    eps: sgd_eps,
                    delta: sgd_delta}
            else: 
                feed_dict = {
                    x: random_batch_data,
                    z: random_batch_z,
                    x_test: random_batch_data_test,
                    avg_dist_x_to_x_test: avg_dist_x_to_x_test_precomputed_}


            # RUN OPTIMIZATION STEP.
            sess.run(g_optim, feed_dict)

            # Occasionally update learning rate.
            if step % lr_update_step == lr_update_step - 1:
                _, lr_ = sess.run([lr_update, lr])
                print('Updated learning rate to {}'.format(lr_))

            ###################################################################
            # logging()
            # Occasionally log/plot results.
            if step % log_step == 0 and step > 0:
                # Read off from graph.
                cmd_, cmd_k_terms_, mmd_, loss1_, loss2_, summary_result = \
                    sess.run(
                        [cmd, cmd_k_terms, mmd, loss1, loss2, summary_op],
                        feed_dict)
                g_readonly_ = sess.run(
                    g_readonly,
                    {z_readonly: get_random_z(data_num, z_dim, for_training=False)})
                g_batch_ = g_readonly_[np.random.randint(0, data_num, batch_size)]
                g_full_ = g_readonly_
                print(('CMD_GAN. Iter: {}\n  cmd: {:.4f}, mmd: {:.4f} '
                       'loss1: {:.4f}, loss2: {:.4f}').format(
                           step, cmd_, mmd_, loss1_, loss2_))

                # TEST joint moment discrepancy.
                md_batch = compute_noncentral_moment_discrepancy(
                    random_batch_data, g_batch_, k_moments=k_moments,
                    cmd_span_const=cmd_span_const, batch_id=_batch_id,
                    fbo_noisy_moments=fixed_batches_onetime_noisy_moments)
                jmd_batch = compute_joint_moment_discrepancy(
                    random_batch_data, g_batch_, k_moments=k_moments,
                    cmd_span_const=cmd_span_const, batch_id=_batch_id,
                    fbo_noisy_moments=fixed_batches_onetime_noisy_moments,
                    fbo_noisy_jmoments=fixed_batches_onetime_noisy_jmoments)
                print('MD_batch: {:.4f}'.format(md_batch))
                print('JMD_batch: {:.4f}'.format(jmd_batch))


                # TODO: DIAGNOSE NaNs.
                if np.isnan(mmd_):
                    pdb.set_trace()

                ###############################################################
                # Unormalize data and simulations for all logs and plots.
                g_batch_unnormed = unnormalize(
                    g_batch_, data_raw_mean, data_raw_std)
                g_full_unnormed = unnormalize(
                    g_full_, data_raw_mean, data_raw_std)
                data_unnormed = unnormalize(
                    data, data_raw_mean, data_raw_std)
                data_test_unnormed = unnormalize(
                    data_test, data_raw_mean, data_raw_std)

                # Compute disclosure risk.
                (sensitivity, precision, false_positive_rate, tp, fn, fp,
                 tn) = evaluate_presence_risk(
                     data_unnormed, data_test_unnormed, g_full_unnormed)
                     #ball_radius=avg_dist_x_to_x_test_precomputed_)
                sens_minus_fpr = sensitivity - false_positive_rate
                print('  Sens={:.4f}, Prec={:.4f}, Fpr: {:.4f}, '
                      'tp: {}, fn: {}, fp: {}, tn: {}'.format(
                          sensitivity, precision, false_positive_rate, tp, fn,
                          fp, tn))

                # Add presence discloser stats to summaries.
                summary_writer.add_summary(summary_result, step)
                add_nongraph_summary_items(
                    summary_writer, step,
                    {'misc/sensitivity': sensitivity,
                     'misc/false_positive_rate': false_positive_rate,
                     'misc/sens_minus_fpr': sens_minus_fpr,
                     'misc/precision': precision})

                ###############################################################
                # Save checkpoint.
                saver.save(
                    sess,
                    os.path.join(log_dir, 'checkpoints', model_type),
                    global_step=step)

                # Save generated data to file.
                np.save(os.path.join(g_out_dir, 'g_out_{}.npy'.format(step)),
                        g_full_unnormed)
                with open(g_out_file, 'a') as f:
                    f.write(str(g_full_unnormed) + '\n')

                # Print time performance.
                if step % (10 * log_step) == 0 and step > 0:
                    elapsed_time = time() - start_time
                    time_per_iter = elapsed_time / step
                    total_est = elapsed_time / step * max_step
                    m, s = divmod(total_est, 60)
                    h, m = divmod(m, 60)
                    total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
                    print ('  Time (s): {:.2f}, time/iter: {:.4f},'
                           ' Total est.: {}').format(
                               elapsed_time, time_per_iter, total_est_str)

                    print '  Save tag: {}'.format(save_tag)


                ################################################################
                # PLOT data and simulations.

                if out_dim == 1:
                    fig, ax = plt.subplots()
                    ax.hist(data, normed=True, bins=30, color='gray', alpha=0.3,
                            label='data')
                    #ax.hist(data_test, normed=True, bins=30, color='red', alpha=0.3,
                    #        label='test')
                    #ax.hist(g_batch_, normed=True, bins=30, color='orange', alpha=0.3,
                    #        label='g_batch')
                    ax.hist(g_full_, normed=True, bins=30, color='blue', alpha=0.2,
                            label='g_full_readonly')

                    plt.legend()
                    plt.savefig(os.path.join(plot_dir, '{}.png'.format(step)))
                    plt.close(fig)
                elif out_dim == 2:
                    #fig, ax = plt.subplots()
                    #ax.scatter(*zip(*data_unnormed), color='gray', alpha=0.2, label='data')
                    #ax.scatter(*zip(*data_test_unnormed), color='red', alpha=0.2, label='test')
                    #ax.scatter(*zip(*g_full_unnormed), color='green', alpha=0.2, label='sim')
                    #ax.legend()
                    #ax.set_title(tag)
                    #plt.savefig(os.path.join(
                    #    plot_dir, '{}.png'.format(step)))
                    #plt.close(fig)
                    d_x = data_unnormed[:, 0]
                    d_y = data_unnormed[:, 1]
                    g_x = g_full_unnormed[:, 0]
                    g_y = g_full_unnormed[:, 1]

                    fig = plt.figure()
                    gs = GridSpec(4, 4)
                    ax_joint = fig.add_subplot(gs[1:4, 0:3])
                    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
                    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

                    ax_joint.scatter(*zip(*data_unnormed), color='gray',
                                     alpha=0.2, label='data')
                    ax_joint.scatter(*zip(*g_full_unnormed), color='green',
                                     alpha=0.2, label='sim')
                    bins_x = np.arange(np.min([np.min(d_x), np.min(g_x)]),
                                       np.max([np.max(d_x), np.max(g_x)]), 0.2)
                    bins_y = np.arange(np.min([np.min(d_y), np.min(g_y)]),
                                       np.max([np.max(d_y), np.max(g_y)]), 0.2)
                    #bins_y = np.arange(np.min(d_y), np.max(d_y), 0.2)
                    ax_marg_x.hist([d_x, g_x], bins=bins_x, alpha=0.2,
                                   color=['gray', 'green'], label=['data', 'gen'],
                                   normed=True)
                    ax_marg_y.hist([d_y, g_y], bins=bins_y,alpha=0.2, 
                                   color=['gray', 'green'], label=['data', 'gen'],
                                   normed=True, orientation='horizontal')
                    ax_joint.legend()
                    ax_marg_x.legend()
                    ax_marg_y.legend()
                    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
                    plt.suptitle(tag)
                    plt.savefig(os.path.join(
                        plot_dir, '{}.png'.format(step)))
                    plt.close()
                    
                
                # PLOT moment diagnostics.
                if data_dim <= 2:
                    normed_moments_gens = compute_moments(
                        g_full_, k_moments=k_moments+1)
                    empirical_moments_gens[step / log_step] = normed_moments_gens

                    # Define colormap used for plotting.
                    cmap = plt.cm.get_cmap('cool', k_moments+1)

                    if data_dim == 1:
                        ###########################################################
                        # Plot empirical moments throughout training.
                        fig, (ax_data, ax_gens) = plt.subplots(2, 1)
                        for i in range(k_moments+1):
                            ax_data.axhline(y=normed_moments_data[i],
                                            label='m{}'.format(i+1), c=cmap(i))
                            ax_gens.plot(empirical_moments_gens[:step/log_step, i],
                                         label='m{}'.format(i+1), c=cmap(i), alpha=0.8)
                        ax_data.set_ylim(min(normed_moments_data)-0.5, max(normed_moments_data)+0.5)
                        ax_gens.set_xlabel('Empirical moments, gens')
                        ax_data.set_xlabel('Empirical moments, data')
                        ax_gens.legend()
                        ax_data.legend()
                        plt.suptitle('{}, empirical moments, k={}'.format(tag, k_moments))
                        plt.tight_layout()
                        plt.savefig(os.path.join(
                            plot_dir, 'empirical_moments.png'))
                        plt.close(fig)


                    # FIGURE OUT TO PLOT RELATIVE ERROR
                    relative_error_of_moments_test = (
                        norm(np.array(normed_moments_data_test) -
                             np.array(normed_moments_data), axis=1) /
                        norm(np.array(normed_moments_data), axis=1))
                    relative_error_of_moments_gens = (
                        norm(np.array(normed_moments_gens) -
                             np.array(normed_moments_data), axis=1) /
                        norm(np.array(normed_moments_data), axis=1))

                    relative_error_of_moments_test[nmd_zero_indices] = 0.0
                    relative_error_of_moments_gens[nmd_zero_indices] = 0.0
                    reom[step / log_step] = relative_error_of_moments_gens

                    if data_dim <= 2:
                        print('    RELATIVE_TEST: {}'.format(list(
                            np.round(relative_error_of_moments_test, 2))))
                        print('    RELATIVE_GENS: {}'.format(list(
                            np.round(relative_error_of_moments_gens, 2))))

                    # For plotting, zero-out moments that are likely zero, so
                    # their relative values don't dominate the plot.
                    reom_trim_level = np.max(np.abs(reom[:, :k_moments]))
                    reom_trimmed = np.copy(reom)
                    reom_trimmed[
                        np.where(reom_trimmed > reom_trim_level)] = \
                                2 * reom_trim_level
                    reom_trimmed[
                        np.where(reom_trimmed < -reom_trim_level)] = \
                                -2 * reom_trim_level
                    fig, ax = plt.subplots()
                    for i in range(k_moments+1):
                        ax.plot(reom[:step/log_step, i],
                                label='m{}'.format(i+1), c=cmap(i))
                    #ax.set_ylim((-2 * reom_trim_level, 2 * reom_trim_level))
                    ax.set_ylim((-2, 2))
                    ax.legend()
                    plt.suptitle('{}, relative errors of moments, k={}'.format(
                        tag, k_moments))
                    plt.savefig(os.path.join(
                        plot_dir, 'reom.png'))
                    plt.close(fig)

                    # Print normed moments to console.
                    if data_dim == 1:
                        print('    data_normed moments: {}'.format(
                            normed_moments_data))
                        print('    test_normed moments: {}'.format(
                            normed_moments_data_test))
                        print('    gens_normed moments: {}'.format(
                            normed_moments_gens))



if __name__ == "__main__":
    main()
