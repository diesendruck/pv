""" This script runs several privacy-oriented MMD-GAN, CMD, and MMD+AE style
    generative models.
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
from scipy.spatial.distance import pdist
from scipy.stats import truncnorm

import tensorflow as tf
layers = tf.layers
#from tensorflow.python.framework import ops
#from tensorflow.python.ops import math_ops
#from tensorflow.python.keras._impl.keras import constraints
# print(constraints)
# >> /usr/local/lib/python2.7/dist-packages/tensorflow/python/keras/_impl/keras

sys.path.append('/home/maurice/mmd')
from mmd_utils import (compute_mmd, compute_kmmd, compute_cmd,
                       compute_noncentral_moment_discrepancy,
                       compute_noncentral_noisy_moment_discrepancy,
                       compute_moments, compute_central_moments,
                       compute_kth_central_moment, MMD_vs_Normal_by_filter,
                       dp_sensitivity_to_expectation)
from local_constraints import (UnitNorm, DivideByMaxNorm, ClipNorm,
                               EdgeIntervalNorm, DivideByMaxThenMinMaxNorm)


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='mmd_gan',
                    choices=['mmd_gan'])
parser.add_argument('--pretrain_steps', type=int, default=10000)
parser.add_argument('--mmd_variation', type=str, default=None,
                    choices=['mmd_to_cmd', ''])
parser.add_argument('--cmd_variation', type=str, default=None,
                    choices=['minus_k_plus_1', 'minus_mmd',
                             'prog_cmd', 'mmd_to_cmd',
                             'noncentral_noisy',
                             'noncentral_onetime_noisy',
                             ''])
parser.add_argument('--ae_variation', type=str, default=None,
                    choices=['pure', 'enc_noise', 'partition_ae_data',
                             'partition_enc_enc', 'subset', 'cmd_k', 'mmd',
                             'cmd_k_minus_k_plus_1'])
parser.add_argument('--noise_type', type=str, default='laplace',
                    choices=['laplace', 'gaussian'])
parser.add_argument('--laplace_eps', type=float, default=2)
parser.add_argument('--data_file', type=str, default='')
parser.add_argument('--data_set', type=str, default='')
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
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop'])
parser.add_argument('--k_moments', type=int, default=2)
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--load_existing', default=False, action='store_true',
                    dest='load_existing')
args = parser.parse_args()
model_type = args.model_type
pretrain_steps = args.pretrain_steps
mmd_variation = args.mmd_variation
cmd_variation = args.cmd_variation
ae_variation = args.ae_variation
laplace_eps = args.laplace_eps
noise_type = args.noise_type
data_file = args.data_file
data_set = args.data_set
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
optimizer = args.optimizer
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
        return truncnorm.rvs(-2, 2, size=[gen_num, z_dim])
        #return np.random.normal(size=[gen_num, z_dim])
    else:
        return truncnorm.rvs(-2, 2, size=[gen_num, z_dim])


def dense(x, width, activation, batch_residual=False, use_bias=True, name=None):
    """Wrapper on fully connected TensorFlow layer."""
    # TODO: Should any of this use bias?
    if batch_residual:
        option = 1
        if option == 1:
            x_ = layers.dense(x, width, activation=activation, use_bias=use_bias)
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
        r = layers.batch_normalization(x_)
    return r


def autoencoder(x, width=3, depth=3, activation=tf.nn.elu, z_dim=3,
                reuse=False, normed_encs=False):
    """Autoencodes input via a bottleneck layer h."""
    out_dim = x.shape[1]
    with tf.variable_scope('encoder', reuse=reuse) as vs_enc:
        x = dense(x, width, activation=activation)

        for idx in range(depth - 1):
            # TODO: Should this use batch resid, and is it defined properly?
            x = dense(x, width, activation=activation, batch_residual=True)

        # NORMED ENCODINGS
        if normed_encs:
            enc_eps = laplace_eps
            enc_delta = 1e-5
            option = 3

            if option == 1:
                # Use sigmoid, and sensitivity is z_dim.
                h_normed = dense(x, z_dim, activation=tf.sigmoid)
                sens1 = z_dim
                sens2 = np.sqrt(z_dim)
                h_noised = h_normed + (
                    np.random.laplace(size=z_dim, loc=0, scale=sens1 / enc_eps)
                    #np.random.normal(size=z_dim, loc=0, scale=np.sqrt(
                    #    2 * np.square(sens2) * np.log(1.25 / enc_delta) / (enc_eps ** 2)))
                    )
            if option == 2:
                # Use steeper sigmoid to push values closer to edges.
                # Sensitivity is z_dim.
                h_ = dense(x, z_dim, activation=None)
                steepness = 15
                h_normed = 1. / (1 + tf.exp(-1. * steepness * h_))
                sens1 = z_dim
                sens2 = np.sqrt(z_dim)
                h_noised = h_normed + (
                    np.random.laplace(size=z_dim, loc=0, scale=sens1 / enc_eps))
            elif option == 3:
                # Use no activation, then DivideByMaxNorm, and sensitivity is
                # 2*sqrt(z_dim).
                h_ = dense(x, z_dim, activation=None)
                h_normed = h_ / tf.reduce_max(h_)
                sens1 = 2 * z_dim
                sens2 = 2 * np.sqrt(z_dim)
                h_noised = h_normed + (
                    np.random.laplace(size=z_dim, loc=0, scale=sens1 / enc_eps))
            h = h_noised
            h_nonoise = h_normed
        else:
            h = dense(x, z_dim, activation=None)

    with tf.variable_scope('decoder', reuse=reuse) as vs_dec:

        x = dense(h, width, activation=activation, use_bias=True,
                  name='hidden')
        if normed_encs:
            x_nonoise = dense(h_nonoise, width, activation=activation,
                              use_bias=True, name='hidden_nonoise')

        for idx in range(depth - 1):
            x = dense(x, width, activation=activation, batch_residual=True)
            if normed_encs:
                x_nonoise = dense(x_nonoise, width, activation=activation,
                                  batch_residual=True)


        ae = dense(x, out_dim, activation=None)
        if normed_encs:
            ae_nonoise = dense(x_nonoise, out_dim, activation=None)


    vars_enc = tf.contrib.framework.get_variables(vs_enc)
    vars_dec = tf.contrib.framework.get_variables(vs_dec)
    if not normed_encs:
        return h, ae, vars_enc, vars_dec
        #return h, _, ae, _, vars_enc, vars_dec
    elif normed_encs:
        return h, h_nonoise, ae, ae_nonoise, vars_enc, vars_dec


def generator(z_in, width=3, depth=3, activation=tf.nn.elu, out_dim=2,
              reuse=False):
    """Decodes. Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse) as vs_g:
        x = dense(z_in, width, activation=activation)

        for idx in range(depth - 1):
            # TODO: Should this use batch resid, and is it defined properly?
            x = dense(x, width, activation=activation, batch_residual=True)

        out = dense(x, out_dim, activation=None)
    vars_g = tf.contrib.framework.get_variables(vs_g)
    return out, vars_g


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


def load_normed_data(data_num, percent_train, data_file=None, data_set=None,
                     save=False):
    """Generates data, and returns it normalized, along with helper objects."""
    # Load data from file.
    if data_file and not save:
        if data_file.endswith('npy'):
            data_raw = np.load(data_file)
        elif data_file.endswith('txt'):
            data_raw = np.loadtxt(open(data_file, 'rb'), delimiter=' ')

    elif not data_file and data_set:
        if data_set == '1d_gaussian':
            data_raw = np.zeros((data_num, 1))
            for i in range(data_num):
                # Pick a Gaussian, then generate from that Gaussian.
                cluster_i = np.random.binomial(1, 0.4)  # NOTE: Setting p=0/p=1 chooses one cluster.
                if cluster_i == 0:
                    data_raw[i] = np.random.normal(0, 2)
                    #data_raw[i] = np.random.gamma(7, 2)
                else:
                    data_raw[i] = np.random.normal(6, 2)

        elif data_set == '2d_gaussian':
            def sample_c1():
                sample = np.random.multivariate_normal(
                    [0., 0.], [[1., 0.], [0., 1.]], 1)
                return sample
            def sample_c2():
                sample = np.random.multivariate_normal(
                    [2., 2.], [[1, 0.5], [0.5, 1]], 1)
                return sample

            data_raw = np.zeros((data_num, 2))
            for i in range(data_num):
                if np.random.binomial(1, 0.3):
                    data_raw[i] = sample_c1()
                else:
                    data_raw[i] = sample_c2()

        elif data_set == '2d_noisy_sin':
            x = np.linspace(0, 10, data_num)
            y = np.sin(x) + np.random.normal(0, 0.5, len(x))
            data_raw = np.hstack((np.expand_dims(x, axis=1),
                                  np.expand_dims(y, axis=1)))
            data_raw = data_raw[np.random.permutation(data_num)]

        else:
            sys.exit('data_set name not recognized')

        if save == True:
            np.save('data_raw_{}.npy'.format(data_set), data_raw)

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


def print_baseline_moment_stats(data, data_num, data_raw_mean, data_raw_std,
                                k_moments):
    """Compute baseline statistics on moments for data set."""
    data_normed = data * data_raw_std + data_raw_mean
    baseline_moments = compute_moments(data_normed, k_moments)
    for j in range(k_moments):
        print 'Moment {}: {}'.format(j+1, baseline_moments[j])


def make_fixed_batches(data, batch_size):
    # Partition data into fixed batches.
    data_dim = data.shape[1]
    fixed_batches = np.array(
        [data[i:i + batch_size] for i in xrange(0, len(data), batch_size)])
    fixed_batches = [b for b in fixed_batches if len(b) == batch_size]
    print('\n\nData size: {}, Batch size: {}, Num batches: {}, '
          'Effective data size: {}\n\n'.format(
              len(data), batch_size, len(fixed_batches),
              batch_size * len(fixed_batches)))
    return np.array(fixed_batches)


def get_noisy_fbmoments(fixed_batches, k_moments, laplace_eps):
    """Computes noisy moments for fixed batches, based on sensitivities of each
       moment across all batches.
    
    Args:
      fixed_batches: NumPy array of fixed batches of inputs, of dimension
        [num_batches, batch_size, input_dim].
      k_moments: Int, describing how many moments to compute.
      laplace_eps: Float, differential privacy epsilon.

    Returns:
      noisy_moments_fb: NumPy array of noisy moments for each batch.
    """
    num_batches = fixed_batches.shape[0]
    batch_size = fixed_batches.shape[1]
    input_dim = fixed_batches.shape[2]

    moments = np.zeros((num_batches, k_moments, input_dim), dtype=np.float32)
    noisy_moments = np.zeros((num_batches, k_moments, input_dim), dtype=np.float32)
    sensitivities = np.zeros((k_moments, input_dim)) 

    # Choose allocation of budget.
    allocation = [1] * k_moments
    eps_ = laplace_eps * (np.array(allocation) / float(np.sum(allocation)))
    assert len(eps_) == k_moments, 'allocation length must match moment num'
    print('Privacy budget, allocation: {}, {}\n'.format(laplace_eps, allocation))

    # Get each batch's moments.
    for i, b in enumerate(fixed_batches):
        for k in range(1, k_moments + 1):
            moments[i, k-1] = np.mean(np.power(b, k), axis=0)

    # For the whole set, compute sensitivity of each moment.
    whole_set = np.concatenate(fixed_batches, axis=0)
    for k in range(1, k_moments + 1):
        mk_sens = (
            (1. / batch_size) *
            np.max(np.power(np.abs(whole_set), k), axis=0))
        sensitivities[k - 1] = mk_sens

    # Compute noisy moments for each batch.
    for i, b in enumerate(fixed_batches):
        for k in range(1, k_moments + 1):
            mk = moments[i, k-1]
            # Sample laplace noise for each dimension of data -- scale
            # param takes vector of laplace scales and outputs
            # corresponding values.
            mk_sens = sensitivities[k - 1]  
            mk_laplace = np.random.laplace(loc=0,
                                           scale=mk_sens/eps_[k-1])
            mk_noisy = mk + mk_laplace
            noisy_moments[i, k-1] = mk_noisy

    # Check that outputs are somewhat close for large eps.
    print ' Sample: RAW moments'
    print moments[:3]
    print ' Sample: NOISY moments'
    print noisy_moments[:3]

    return noisy_moments


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


def build_model_mmd_gan(batch_size, gen_num, data_num, data_test_num, out_dim,
                        z_dim, cmd_span_const):
    """Builds model for MMD as adversary."""
    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
                                  name='x_precompute')
    x_test_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
                                       name='x_test_precompute')
    avg_dist_x_to_x_test_precomputed, distances_xt_xp = \
        avg_nearest_neighbor_distance(x_precompute, x_test_precompute,
                                      flag='xt_to_xp')

    # Regular training placeholders.
    x = tf.placeholder(tf.float32, [batch_size, out_dim], name='x')
    x_readonly = tf.placeholder(tf.float32, [None, out_dim], name='x_readonly')
    z = tf.placeholder(tf.float32, [gen_num, z_dim], name='z')
    z_readonly = tf.placeholder(tf.float32, [None, z_dim], name='z_readonly')
    x_test = tf.placeholder(tf.float32, [batch_size, out_dim], name='x_test')
    avg_dist_x_to_x_test = tf.placeholder(tf.float32, shape=(),
                                          name='avg_dist_x_to_x_test')
    batch_id = tf.placeholder(tf.int32, shape=(), name='batch_id')
    noisy_moments_fb = tf.placeholder(tf.float32, [None, k_moments, z_dim],
                                      name='noisy_moments_fb')

    # Create simulation and set up and autoencoder inputs.
    g, g_vars = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim)
    g_readonly, _ = generator(
        z_readonly, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)

    # Compute with autoencoder.
    # Autoencoding of data and generated batches.
    ae_in = tf.concat([x, g], 0)
    h_out, ae_out, enc_vars, dec_vars = autoencoder(
        ae_in, width=width, depth=depth, activation=activation, z_dim=z_dim,
        reuse=False)
    enc_x, enc_g = tf.split(h_out, [batch_size, gen_num])
    ae_x, ae_g = tf.split(ae_out, [batch_size, gen_num])
    # Autoencoding of readonly requests.
    enc_x_readonly, ae_x_readonly, _, _ = autoencoder(
        x_readonly, width=width, depth=depth, activation=activation,
        z_dim=z_dim, reuse=True)
    enc_g_readonly, ae_g_readonly, _, _ = autoencoder(
        g_readonly, width=width, depth=depth, activation=activation,
        z_dim=z_dim, reuse=True)

    # USEFUL COMPONENTS OF LOSSES
    # How close are encodings to Gaussian distribution?
    enc_norm_loss = MMD_vs_Normal_by_filter(
        enc_x, np.ones([batch_size, 1], dtype=np.float32), sigmas=[0.1, 1., 2.])
    # Compute autoencoder loss on both data and generations.
    #ae_loss = tf.reduce_mean(tf.square(ae_out - ae_in))
    ae_loss = tf.reduce_mean(tf.square(ae_x - x))
    # Compute MMD between encodings of data and encodings of generated.
    mmd_on_encodings = compute_mmd(enc_x, enc_g, sigma_list=[0.1, 1, 2],
                                   use_tf=True, slim_output=True)

    # DESCRIPTIVE LOSSES NOT USED IN TRAINING.
    # Compare distances between data, heldouts, and gen.
    avg_dist_g_to_x, distances_g_x = avg_nearest_neighbor_distance(g, x, flag='g_to_x')
    avg_dist_x_to_g, distances_x_g = avg_nearest_neighbor_distance(
        x, g, flag='x_to_g')
    avg_dist_x_test_to_g, distances_xt_g = avg_nearest_neighbor_distance(
        x_test, g, flag='xt_to_g')
    loss1 = avg_dist_x_to_x_test - avg_dist_x_to_g
    loss2 = avg_dist_x_test_to_g - avg_dist_x_to_g
    # Define MMD and CMD for reporting.
    mmd = compute_mmd(x, g, use_tf=True, slim_output=True)
    cmd = compute_cmd(x, g, k_moments=k_moments, use_tf=True, 
                      cmd_span_const=cmd_span_const)

    # Optimization losses.
    d_loss = ae_loss + 0.1 * enc_norm_loss - mmd_on_encodings
    #g_loss = mmd_on_encodings
    g_loss = compute_noncentral_moment_discrepancy(
        enc_x, enc_g, k_moments=k_moments, use_tf=True,
        cmd_span_const=cmd_span_const, batch_id=batch_id,
        fbo_noisy_moments=noisy_moments_fb)

    # Optimization setups.
    lr = tf.Variable(learning_rate, name='lr', trainable=False)
    lr_update = tf.assign(lr, tf.maximum(lr * 0.8, 1e-8), name='lr_update')
    if optimizer == 'adagrad':
        ae_opt = tf.train.AdagradOptimizer(lr)
        d_opt = tf.train.AdagradOptimizer(lr)
        g_opt = tf.train.AdagradOptimizer(lr)
    elif optimizer == 'adam':
        ae_opt = tf.train.AdamOptimizer(lr)
        d_opt = tf.train.AdamOptimizer(lr)
        g_opt = tf.train.AdamOptimizer(lr)
    elif optimizer == 'rmsprop':
        ae_opt = tf.train.RMSPropOptimizer(lr)
        d_opt = tf.train.RMSPropOptimizer(lr)
        g_opt = tf.train.RMSPropOptimizer(lr)
    else:
        ae_opt = tf.train.GradientDescentOptimizer(lr)
        d_opt = tf.train.GradientDescentOptimizer(lr)
        g_opt = tf.train.GradientDescentOptimizer(lr)
    # Define optim nodes.
    clip = 0
    if clip:
        enc_grads_, enc_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=enc_vars))
        dec_grads_, dec_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=dec_vars))
        enc_grads_clipped_ = tuple(
            [tf.clip_by_value(grad, -0.01, 0.01) for grad in enc_grads_])
        d_grads_ = enc_grads_clipped_ + dec_grads_
        d_vars_ = enc_vars_ + dec_vars_
        d_optim = d_opt.apply_gradients(zip(d_grads_, d_vars_))
        ae_optim = ae_opt.minimize(ae_loss + 0.1 * enc_norm_loss, var_list=d_vars)
    else:
        d_optim = d_opt.minimize(d_loss, var_list=enc_vars + dec_vars)
        ae_optim = ae_opt.minimize(ae_loss + 0.1 * enc_norm_loss, var_list=enc_vars + dec_vars)
    g_optim = g_opt.minimize(g_loss, var_list=g_vars)

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/ae_loss", ae_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/cmd", cmd),
	#tf.summary.scalar("loss/g_loss", g_loss),
	tf.summary.scalar("loss/d_loss", d_loss),
	tf.summary.scalar("misc/lr", lr),
    ])

    return (x, ae_x, x_readonly, enc_x_readonly, enc_g_readonly,
            ae_x_readonly, ae_g_readonly, z, z_readonly, x_test,
            x_precompute, x_test_precompute, avg_dist_x_to_x_test,
            avg_dist_x_to_x_test_precomputed, distances_xt_xp,
            g, g_readonly, ae_loss, enc_norm_loss, d_loss,
            mmd, mmd_on_encodings, cmd, loss1, loss2, lr_update, d_optim,
            g_optim, ae_optim, summary_op, batch_id, noisy_moments_fb)


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
    if data_file:
        (data, data_test, data_num, data_test_num, out_dim, data_raw_mean,
         data_raw_std) = load_normed_data(data_num, percent_train,
                                          data_file=data_file)
    else:
        (data, data_test, data_num, data_test_num, out_dim, data_raw_mean,
         data_raw_std) = load_normed_data(data_num, percent_train,
                                          data_set=data_set, save=True)
        print('saved new data file') 
    data_dim = data.shape[1]
    normed_moments_data = compute_moments(data, k_moments=k_moments+1)
    normed_moments_data_test = compute_moments(data_test, k_moments=k_moments+1)
    nmd_zero_indices = np.argwhere(
        norm(np.array(normed_moments_data), axis=1) < 0.1)

    # Compute baseline statistics on moments for data set.
    print_baseline_moment_stats(data, data_num, data_raw_mean, data_raw_std,
                                k_moments)

    # Compute sensitivities for moments, based on fixed batches.
    fixed_batches = make_fixed_batches(data, batch_size)

    # Get compact interval bounds for CMD computations.
    cmd_a = np.min(data, axis=0)
    cmd_b = np.max(data, axis=0)
    cmd_span_const = 1.0 / np.max(pdist(data))
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

    # build_model()
    # Build model.
    (x, ae_x, x_readonly, enc_x_readonly, enc_g_readonly, ae_x_readonly,
     ae_g_readonly, z, z_readonly, x_test, x_precompute, x_test_precompute,
     avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
     distances_xt_xp, g, g_readonly, ae_loss,
     enc_norm_loss, d_loss, mmd, mmd_on_encodings, cmd, loss1, loss2,
     lr_update, d_optim, g_optim, ae_optim, summary_op, batch_id,
     noisy_moments_fb) = \
         build_model_mmd_gan(
             batch_size, gen_num, data_num, data_test_num, out_dim, z_dim,
             cmd_span_const)

    ###########################################################################
    # Start session.
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=sess_config) as sess:
        saver, summary_writer = prepare_logging(log_dir, checkpoint_dir, sess)
        sess.run(init_op)

        # Load existing model.
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

            # Set up input procedure. Option 1: Random batch selection.
            # Option 2: Fixed batch selection.
            batch_selection_option = 2
            if batch_selection_option == 1:
                random_batch_data = np.array(
                    [data[d] for d in np.random.choice(len(data), batch_size)])
                _batch_id = None
            elif batch_selection_option == 2:
                _epoch, _batch_id = np.divmod(step, len(fixed_batches))
                if _batch_id == 0:
                    print 'Epoch {}'.format(_epoch)
                random_batch_data = fixed_batches[_batch_id]

            # Fetch test data and z.
            random_batch_data_test = np.array(
                [data_test[d] for d in np.random.choice(
                    len(data_test), batch_size)])
            random_batch_z = get_random_z(gen_num, z_dim)

            # Update shared dict for chosen model.
            feed_dict = {
                z: random_batch_z,
                x: random_batch_data,
                x_test: random_batch_data_test,
                x_readonly: data,
                avg_dist_x_to_x_test: avg_dist_x_to_x_test_precomputed_}

            # Run optimization step.
            # Pretrain autoencoder to have Gaussian encodings. Then one-time,
            # get encoding moments, sensitivities, and add noise. Finally,
            # Proceed to generator training, passing in batch_id and noisy
            # moment info.
            if step < pretrain_steps:
                sess.run([ae_optim], feed_dict)
            elif step == pretrain_steps:
                # Encode data, keeping shape [num_batches, batch_size, z_dim].
                enc_all_fb = np.array([
                    sess.run(enc_x_readonly, {x_readonly: b}) for b in fixed_batches]) 
                noisy_moments_fb_enc = get_noisy_fbmoments(
                    fixed_batches, k_moments, laplace_eps)
            else:
                feed_dict.update({
                    batch_id: _batch_id,
                    noisy_moments_fb: noisy_moments_fb_enc})
                sess.run([g_optim], feed_dict)


            # Occasionally update learning rate.
            if step % lr_update_step == lr_update_step - 1:
                sess.run(lr_update)

            ###################################################################
            # log()
            # Occasionally log/plot results.
            if step % log_step == 0 and step > 0:
                # Read off from graph.
                (d_loss_,
                 ae_loss_,
                 enc_norm_loss_,
                 mmd_on_encodings_,
                 mmd_,
                 loss1_,
                 loss2_,
                 summary_result) = sess.run(
                    [d_loss, ae_loss, enc_norm_loss, mmd_on_encodings, mmd,
                     loss1, loss2, summary_op],
                    feed_dict)
                print(('\nMMD_GAN. Iter: {}\n  d_loss: {:.4f}, ae_loss: {:.4f}, '
                       'enc_norm_loss: {:.4f}, mmd_on_encodings: {:.4f}, '
                       'mmd: {:.4f}, loss1: {:.4f}, loss2: {:.4f}').format(
                           step, d_loss_, ae_loss_, enc_norm_loss_,
                           mmd_on_encodings_, mmd_, loss1_, loss2_))
                (g_full_,
                 ae_x_full_,
                 enc_x_full_,
                 enc_g_full_) = sess.run(
                     [g_readonly, ae_x_readonly, enc_x_readonly,
                      enc_g_readonly],
                     feed_dict={
                         z_readonly: get_random_z(data_num, z_dim),
                         x_readonly: data})
                # Test this step's batch.
                feed_dict.update({x: random_batch_data})
                ae_x_batch_, g_batch_ = sess.run([ae_x, g], feed_dict)

                # TODO: DIAGNOSE NaNs.
                if np.isnan(mmd_):
                    pdb.set_trace()

                ###############################################################
                # Unormalize data and simulations for all logs and plots.
                drm, drs = data_raw_mean, data_raw_std
                [g_batch_unnormed,
                 g_full_unnormed,
                 ae_full_unnormed,
                 data_unnormed,
                 data_test_unnormed] = \
                    [unnormalize(s, drm, drs) for s in [g_batch_, g_full_,
                                                        ae_x_full_, data,
                                                        data_test]]
                #g_batch_unnormed = unnormalize(g_batch_, drm, drs)
                #g_full_unnormed = unnormalize(g_full_, drm, drs)
                #ae_full_unnormed = unnormalize(ae_x_full_, drm, drs)
                #data_unnormed = unnormalize(data, drm, drs)
                #data_test_unnormed = unnormalize(data_test, drm, drs)

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

                    # PLOT RAW UNNORMED VERSIONS. #############################
                    ax.hist(data, normed=True, bins=30, color='gray', alpha=0.3,
                            label='data')
                    #ax.hist(data_test, normed=True, bins=30, color='red', alpha=0.3,
                    #        label='test')
                    ax.hist(g_batch_, normed=True, bins=30, color='orange', alpha=0.3,
                            label='g_batch')
                    ax.hist(g_full_, normed=True, bins=30, color='blue', alpha=0.2,
                            label='g_full_readonly')

                    plt.legend()
                    plt.savefig(os.path.join(plot_dir, '{}.png'.format(step)))
                    plt.close(fig)
                elif out_dim == 2:
                    fig, ax = plt.subplots()
                    ax.scatter(*zip(*data_unnormed), color='gray', alpha=0.2, label='data')
                    #ax.scatter(*zip(*data_test_unnormed), color='red', alpha=0.2, label='test')
                    ax.scatter(*zip(*ae_full_unnormed[:1000]), color='blue', alpha=0.1, label='ae_data')
                    ax.scatter(*zip(*g_full_unnormed[:1000]), color='green', alpha=0.2, label='sim')
                    ax.legend()
                    ax.set_title(tag)
                    plt.savefig(os.path.join(
                        plot_dir, '{}.png'.format(step)))
                    plt.close(fig)
                
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

                    ###########################################################
                    # Plot encoding space.
                    if enc_x_full_.shape[1] == 2:
                        if step < pretrain_steps:
                            fig, ax = plt.subplots()
                            ax.scatter(*zip(*enc_x_full_), color='gray',
                                       alpha=0.2, label='enc_x')
                            #ax.scatter(*zip(*enc_g_full_[:1000]), color='green',
                            #           alpha=0.2, label='enc_g')
                            ax.legend()
                            ax.set_title(tag)
                            plt.savefig(os.path.join(
                                plot_dir, 'enc_{}.png'.format(step)))
                            plt.close(fig)
                        else:
                            fig, ax = plt.subplots()
                            ax.scatter(*zip(*enc_x_full_), color='gray',
                                       alpha=0.2, label='enc_x')
                            ax.scatter(*zip(*enc_g_full_[:1000]), color='green',
                                       alpha=0.2, label='enc_g')
                            ax.legend()
                            ax.set_title(tag)
                            plt.savefig(os.path.join(
                                plot_dir, 'enc_{}.png'.format(step)))
                            plt.close(fig)


if __name__ == "__main__":
    main()
