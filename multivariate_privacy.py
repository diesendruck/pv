import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from numpy.linalg import norm
import os
import pdb
import shutil
import sys
sys.path.append('/home/maurice/mmd')
import tensorflow as tf
layers = tf.layers

from tensorflow import keras
from mmd_utils import compute_mmd, compute_kmmd, compute_cmd, compute_moments, MMD_vs_Normal_by_filter
from scipy.stats import multivariate_normal, shapiro
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops



""" This script runs several privacy-oriented MMD-GAN, CMD, and MMD+AE style
    generative models. 
"""


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='mmd_gan',
                    choices=['ae_base', 'mmd_gan', 'mmd_gan_simple',
                             'kmmd_gan', 'cmd_gan'])
parser.add_argument('--mmd_variation', type=str, default=None,
                     choices=['mmd_to_cmd'])
parser.add_argument('--cmd_variation', type=str, default=None,
                     choices=['minus_k_plus_1', 'minus_mmd',
                              'prog_cmd', 'mmd_to_cmd', 'fractional'])
parser.add_argument('--ae_variation', type=str, default=None,
                     choices=['pure', 'laplace', 'partition_ae_data',
                              'partition_enc_enc', 'subset', 'cmd_k', 'mmd',
                              'cmd_k_minus_k_plus_1'])
parser.add_argument('--data_num', type=int, default=10000)
parser.add_argument('--data_dimension', type=int, default=1)
parser.add_argument('--percent_train', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--gen_num', type=int, default=200)
parser.add_argument('--width', type=int, default=20,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=10,
                    help='num of generator layers')
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--log_step', type=int, default=1000)
parser.add_argument('--max_step', type=int, default=100000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_update_step', type=int, default=500000)
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop'])
parser.add_argument('--data_file', type=str, default='gp_data.txt')
parser.add_argument('--k_moments', type=int, default=2)
parser.add_argument('--kernel_choice', type=str, default='rbf_taylor',
                    choices=['poly', 'rbf_taylor'])
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--load_existing', default=False, action='store_true',
                    dest='load_existing')
args = parser.parse_args()
model_type = args.model_type
mmd_variation = args.mmd_variation
cmd_variation = args.cmd_variation
ae_variation = args.ae_variation
data_num = args.data_num
data_dimension = args.data_dimension
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
data_file = args.data_file
k_moments = args.k_moments
kernel_choice = args.kernel_choice
sigma = args.sigma
tag = args.tag
load_existing = args.load_existing

activation = tf.nn.elu
data_num = int(percent_train * data_num)  # Update based on % train.


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    return np.random.normal(size=[gen_num, z_dim])


def dense(x, width, activation, batch_residual=False, name=None):
    if not batch_residual:
        x_ = layers.dense(x, width, activation=activation, name=name)
        r = layers.batch_normalization(x_)
        return r
    else:
        # TODO: Should this use bias?
        x_ = layers.dense(x, width, activation=activation, use_bias=True)
        r = layers.batch_normalization(x_) + x
        return r


def autoencoder(x, width=3, depth=3, activation=tf.nn.elu, z_dim=3,
        reuse=False, normed_weights=False):
    """Autoencodes input via a bottleneck layer h."""
    out_dim = x.shape[1]
    with tf.variable_scope('encoder', reuse=reuse) as vs_enc:
        x = dense(x, width, activation=activation)

        for idx in range(depth - 1):
            # TODO: Should this use batch resid, and is it defined properly?
            x = dense(x, width, activation=activation, batch_residual=True)

        # TODO: Figure out whether weights should be normed.
        # For the Laplace noise, use normed_weights and no bias. For all others,
        #   use standard dense layer.
        #if normed_weights:
        #    h = layers.dense(x, z_dim, activation=None, use_bias=False,
        #        name='hidden_nw',
        #        )
        #        #kernel_constraint=keras.constraints.unit_norm(axis=None))
        #        #kernel_constraint=keras.constraints.max_norm(1, axis=None))
        #else:
        #    h = dense(x, z_dim, activation=None, name='hidden')
        h = dense(x, z_dim, activation=None)

    with tf.variable_scope('decoder', reuse=reuse) as vs_dec:

        if normed_weights:
            x = layers.dense(h, width, activation=activation, name='hidden_nw',
                )
                #kernel_constraint=keras.constraints.unit_norm(axis=None))
                #kernel_constraint=keras.constraints.max_norm(1, axis=None))
        else:
            x = dense(h, width, activation=activation, name='hidden')
        #x = dense(h, width, activation=activation)

        for idx in range(depth - 1):
            # TODO: Should this use batch resid, and is it defined properly?
            x = dense(x, width, activation=activation, batch_residual=True)

        ae = dense(x, out_dim, activation=None)

    vars_enc = tf.contrib.framework.get_variables(vs_enc)
    vars_dec = tf.contrib.framework.get_variables(vs_dec)
    return h, ae, vars_enc, vars_dec


def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=2,
        reuse=False):
    """Decodes. Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse) as vs_g:
        x = dense(z, width, activation=activation)

        for idx in range(depth - 1):
            # TODO: Should this use batch resid, and is it defined properly?
            x = dense(x, width, activation=activation, batch_residual=True)

        out = dense(x, out_dim, activation=None)
    vars_g = tf.contrib.framework.get_variables(vs_g)
    return out, vars_g


def load_checkpoint(saver, sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    print("     {}".format(checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        #counter = int(''.join([i for i in ckpt_name if i.isdigit()]))
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def load_data(data_num, percent_train):
    """Generates data, and returns it normalized, along with helper objects.
    """
    # Load data.
    if data_file:
        if data_file.endswith('npy'):
            data_raw = np.load(data_file)
            data_raw = data_raw[:10000]
            print('Using first 10000 from {}.'.format(data_file))
        elif data_file.endswith('txt'):
            data_raw = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
            data_raw = np.random.permutation(data_raw)
    else:
        if data_dimension == 1:
            data_raw = np.zeros((data_num, 1))
            for i in range(data_num):
                # Pick a Gaussian, then generate from that Gaussian.
                cluster_i = np.random.binomial(1, 0.4)  # NOTE: Setting p=0/p=1 chooses one cluster.
                if cluster_i == 0:
                    data_raw[i] = np.random.normal(0, 2)
                    #data_raw[i] = np.random.gamma(7, 2)
                else:
                    data_raw[i] = np.random.normal(6, 2)
        elif data_dimension == 2:
            def sample_c1():
                sample = np.random.multivariate_normal(
                    [0., 0.], [[1., 0.], [0., 1.]], 1)
                return sample
            def sample_c2():
                sample = np.random.multivariate_normal(
                    [4., 4.], [[1, 0.5], [0.5, 1]], 1)
                return sample
            data_raw = np.zeros((data_num, 2))
            for i in range(data_num):
                # Pick a Gaussian, then generate from that Gaussian.
                cluster_i = np.random.binomial(1, 0.3)
                if cluster_i == 0:
                    s = sample_c1()
                    data_raw[i] = s
                else:
                    s = sample_c2()
                    data_raw[i] = s

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


def unnormalize(data_normed, data_raw_mean, data_raw_std):
    return data_normed * data_raw_std + data_raw_mean


def prepare_dirs(load_existing):
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
        smallest_dist_negative, _ =  tf.nn.top_k(distances_negative, name=flag)
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


def build_model_ae_base(batch_size, data_num, data_test_num, gen_num, out_dim,
        z_dim, cmd_a, cmd_b):
    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_precompute')
    x_test_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_test_precompute')
    avg_dist_x_to_x_test_precomputed, distances_xt_xp = \
        avg_nearest_neighbor_distance(x_precompute, x_test_precompute)

    # Regular training placeholders.
    x = tf.placeholder(tf.float32, [batch_size, out_dim], name='x')
    x_full = tf.placeholder(tf.float32, [data_num, out_dim], name='x_full')
    x_test = tf.placeholder(tf.float32, [batch_size, out_dim], name='x_test')
    avg_dist_x_to_x_test = tf.placeholder(tf.float32, shape=(),
        name='avg_dist_x_to_x_test')

    # Autoencoder.
    if ae_variation == 'laplace':
        enc_x, ae_x, enc_vars, dec_vars = autoencoder(x,
            width=width, depth=depth, activation=activation, z_dim=z_dim,
            reuse=False, normed_weights=True)  # Variation uses normed_weights.
        enc_x_full, ae_x_full, _, _ = autoencoder(x_full,
            width=width, depth=depth, activation=activation, z_dim=z_dim,
            reuse=True, normed_weights=True)
    else:
        enc_x, ae_x, enc_vars, dec_vars = autoencoder(x,
            width=width, depth=depth, activation=activation, z_dim=z_dim,
            reuse=False)
        enc_x_full, ae_x_full, _, _ = autoencoder(x_full,
            width=width, depth=depth, activation=activation, z_dim=z_dim,
            reuse=True)

    ae_loss = tf.reduce_mean(tf.square(ae_x - x))
    enc_norm_loss = MMD_vs_Normal_by_filter(enc_x, np.ones([batch_size, 1], dtype=np.float32))
    g = ae_x
    g_full = ae_x_full

    # Do weight management. For non laplace settings, it goes unused.
    eps = 5.0
    delta = 1e-5
    h_weights = [v for v in tf.trainable_variables() if 'hidden' in v.name][0]
    num_weights = np.prod(h_weights.shape.as_list())
    gaussian_noise = np.random.normal(size=h_weights.shape.as_list(), loc=0,
        scale=np.sqrt(8 * num_weights * np.log(1.25 / delta) / (eps**2)))
    laplace_noise = np.random.laplace(size=h_weights.shape.as_list(), loc=0,
        #scale=np.sqrt(2 * num_weights / (eps**2)))
        scale=2 * num_weights / eps)
    h_weights_update = tf.assign_add(h_weights, laplace_noise)

    # Compare distances between data, heldouts, and gen.
    avg_dist_g_to_x, distances_g_x = avg_nearest_neighbor_distance(g, x)
    avg_dist_x_to_g, distances_x_g = avg_nearest_neighbor_distance(x, g)
    avg_dist_x_test_to_g, distances_xt_g = avg_nearest_neighbor_distance(x_test, g)
    loss1 = avg_dist_x_to_x_test - avg_dist_x_to_g 
    loss2 = avg_dist_x_test_to_g - avg_dist_x_to_g

    # Define MMD and CMD for reporting.
    mmd = compute_mmd(ae_x, x, use_tf=True, slim_output=True)
    cmd = compute_cmd(ae_x, x, k_moments=k_moments, use_tf=True, cmd_a=cmd_a,
        cmd_b=cmd_b)

    # Set up optimizer, and learning rate.
    lr = tf.Variable(learning_rate, name='lr', trainable=False)
    lr_update = tf.assign(lr, tf.maximum(lr * 0.8, 1e-8), name='lr_update')
    if optimizer == 'adagrad':
        d_opt = tf.train.AdagradOptimizer(lr)
    elif optimizer == 'adam':
        d_opt = tf.train.AdamOptimizer(lr)
    elif optimizer == 'rmsprop':
        d_opt = tf.train.RMSPropOptimizer(lr)
    else:
        d_opt = tf.train.GradientDescentOptimizer(lr)

    # Define privacy mechanism, and ultimately the "d_loss".
    if ae_variation == 'pure':
        ae_base_loss = ae_loss
        d_loss = ae_loss 
    elif ae_variation == 'partition_ae_data':
        # Shuffle, then MMD between autoencodings of first half, and raw data second half.
        permuted_indices = np.random.permutation(batch_size)
        p1_indices = permuted_indices[:batch_size/2]
        p2_indices = permuted_indices[batch_size/2:]
        ae_x_p1 = tf.gather(ae_x, p1_indices)  # Get encodings for one partition.
        x_p2 = tf.gather(x, p2_indices)  # Compare to raw for other partition.
        ae_base_loss = compute_mmd(ae_x_p1, x_p2, use_tf=True, slim_output=True)
        d_loss = 1.5 * ae_loss + 10. * ae_base_loss
    elif ae_variation == 'partition_enc_enc':
        # Shuffle encodings, then MMD between encodings of first and second half.
        permuted_indices = np.random.permutation(batch_size)
        p1_indices = permuted_indices[:batch_size/2]
        p2_indices = permuted_indices[batch_size/2:]
        enc_x_p1 = tf.gather(enc_x, p1_indices) 
        enc_x_p2 = tf.gather(enc_x, p2_indices)
        ae_base_loss = compute_mmd(enc_x_p1, enc_x_p2, use_tf=True,
            slim_output=True)
        d_loss = ae_loss + 10. * ae_base_loss
    elif ae_variation == 'subset':
        # MMD between autoencodings of batch, and batch subset.
        subset_indices = np.random.choice(
            batch_size, int(batch_size * 0.25), replace=False)  
        x_subset = tf.gather(x, subset_indices) 
        _, ae_x_subset, _, _ = autoencoder(x_subset,
            width=width, depth=depth, activation=activation, z_dim=z_dim,
            reuse=True)
        ae_base_loss = compute_mmd(ae_x, x_subset, use_tf=True,
            slim_output=True)
        d_loss = 0.1 * ae_loss + 10. * ae_base_loss
    elif ae_variation == 'mmd':
        # MMD between autoencodings of batch, and batch.
        ae_base_loss = compute_mmd(ae_x, x, use_tf=True, slim_output=True)
        d_loss = ae_loss + 2. * ae_base_loss
    elif ae_variation == 'cmd_k':
        # CMD between batch and autoencodings of batch.
        ae_base_loss = compute_cmd(ae_x, x, k_moments=k_moments, use_tf=True,
            cmd_a=cmd_a, cmd_b=cmd_b)  
        d_loss = ae_loss + 2. * ae_base_loss
    elif ae_variation == 'cmd_k_minus_k_plus_1':
        # Same as above, but with diverging k+1'th moment.
        cmd_k = compute_cmd(ae_x, x, k_moments=k_moments, use_tf=True,
            cmd_a=cmd_a, cmd_b=cmd_b)  
        cmd_k_minus_k_plus_1 = compute_cmd(ae_x, x, k_moments=k_moments+1,
            use_tf=True, cmd_a=cmd_a, cmd_b=cmd_b)
        ae_base_loss = 2 * cmd_k - cmd_k_minus_k_plus_1
        d_loss = ae_loss + 2. * ae_base_loss
    elif ae_variation == 'laplace':
        # Define loss that encourages encodings near standard Gaussian.
        ae_base_loss = ae_loss
        d_loss = ae_loss + 0. * enc_norm_loss
        d_optim = d_opt.minimize(d_loss, var_list=enc_vars + dec_vars)


    # Define optim nodes, optionally clipping encoder gradients.
    clip = 1
    if clip:
        if ae_variation != 'laplace':
            # Clip all weights.
            enc_grads_, enc_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=enc_vars))
            dec_grads_, dec_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=dec_vars))
            enc_grads_clipped_ = tuple(
                [tf.clip_by_value(grad, -0.01, 0.01) for grad in enc_grads_])
            d_grads_ = enc_grads_clipped_ + dec_grads_
            d_vars_ = enc_vars_ + dec_vars_
            d_optim = d_opt.apply_gradients(zip(d_grads_, d_vars_))
        elif ae_variation == 'laplace':
            # Just encoder, grads and vars.
            enc_grads_, enc_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=enc_vars))

            # Decoder grads and vars, split into hidden and non-hidden.
            hidden_vars = [v for v in tf.trainable_variables() if 'hidden' in v.name]
            not_hidden_vars = [v for v in dec_vars if v not in hidden_vars]
            hv_grads_, hv_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=hidden_vars))
            nhv_grads_, nhv_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=not_hidden_vars))

            # Clip only weights of hidden layer, to [-1, 1].
            hv_grads_clipped_ = tuple(
                [tf.clip_by_value(grad, -1., 1.) for grad in hv_grads_])

            # Collect grads, collect vars, in the correct order, and apply them.
            d_grads_ = enc_grads_ + hv_grads_clipped_ + nhv_grads_
            d_vars_ = enc_vars_ + hv_vars_ + nhv_vars_
            d_optim = d_opt.apply_gradients(zip(d_grads_, d_vars_))
    else:
        d_optim = d_opt.minimize(d_loss, var_list=enc_vars + dec_vars)

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/ae_loss", ae_loss),
	tf.summary.scalar("loss/ae_base_loss", ae_base_loss),
	tf.summary.scalar("loss/enc_norm_loss", enc_norm_loss),
	tf.summary.scalar("loss/d_loss", d_loss),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/cmd", cmd),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("misc/lr", lr),
    ])

    return (x, x_full, enc_x_full, x_test, x_precompute, x_test_precompute,
            avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
            distances_xt_xp, g, g_full, ae_loss, ae_base_loss,
            enc_norm_loss, d_loss, mmd, cmd, loss1, loss2, lr_update,
            d_optim, summary_op, h_weights, h_weights_update)


def build_model_mmd_gan(batch_size, gen_num, data_num, data_test_num, out_dim,
        z_dim, cmd_a, cmd_b):
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
    x_full = tf.placeholder(tf.float32, [data_num, out_dim], name='x_full')
    z = tf.placeholder(tf.float32, [gen_num, z_dim], name='z')
    z_full = tf.placeholder(tf.float32, [data_num, z_dim], name='z_full')
    x_test = tf.placeholder(tf.float32, [batch_size, out_dim], name='x_test')
    avg_dist_x_to_x_test = tf.placeholder(tf.float32, shape=(),
        name='avg_dist_x_to_x_test')
    mmd_to_cmd_indicator = tf.placeholder(tf.float32, shape=(),
        name='mmd_to_cmd_indicator')

    # Create simulation and set up and autoencoder inputs.
    g, g_vars = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim)
    g_full, _ = generator(
        z_full, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)
    ae_in = tf.concat([x, g], 0)

    # Compute with autoencoder.
    # Autoencoding of data and generated batches.
    h_out, ae_out, enc_vars, dec_vars = autoencoder(ae_in,
        width=width, depth=depth, activation=activation, z_dim=z_dim,
        reuse=False)
    enc_x, enc_g = tf.split(h_out, [batch_size, gen_num])
    ae_x, ae_g = tf.split(ae_out, [batch_size, gen_num])

    # Autoencoding of full data set.
    enc_x_full, ae_x_full, _, _ = autoencoder(x_full,
        width=width, depth=depth, activation=activation, z_dim=z_dim,
        reuse=True)

    # Compute autoencoder loss on both data and generations.
    ae_loss = tf.reduce_mean(tf.square(ae_out - ae_in))
    # Compute MMD between encodings of data and encodings of generated.
    mmd_on_encodings = compute_mmd(enc_x, enc_g, use_tf=True, slim_output=True)

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
    cmd = compute_cmd(x, g, k_moments=k_moments, use_tf=True, cmd_a=cmd_a,
        cmd_b=cmd_b)

    # Train with MMD, and switch to CMD at inidcator step.
    if mmd_variation == 'mmd_to_cmd':
        mmd_adjusted = (1 - mmd_to_cmd_indicator) * mmd + \
            (mmd_to_cmd_indicator) * cmd
    else:
        mmd_adjusted = mmd_on_encodings

    # Optimization losses.
    d_loss = 0.1 * ae_loss - mmd_adjusted
    g_loss = mmd_adjusted

    lr = tf.Variable(learning_rate, name='lr', trainable=False)
    lr_update = tf.assign(lr, tf.maximum(lr * 0.8, 1e-8), name='lr_update')
    if optimizer == 'adagrad':
        d_opt = tf.train.AdagradOptimizer(lr)
        g_opt = tf.train.AdagradOptimizer(lr)
    elif optimizer == 'adam':
        d_opt = tf.train.AdamOptimizer(lr)
        g_opt = tf.train.AdamOptimizer(lr)
    elif optimizer == 'rmsprop':
        d_opt = tf.train.RMSPropOptimizer(lr)
        g_opt = tf.train.RMSPropOptimizer(lr)
    else:
        d_opt = tf.train.GradientDescentOptimizer(lr)
        g_opt = tf.train.GradientDescentOptimizer(lr)

    # Define optim nodes.
    # Clip encoder gradients.
    clip = 0
    if clip:
        enc_grads_, enc_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=enc_vars))
        dec_grads_, dec_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=dec_vars))
        enc_grads_clipped_ = tuple(
            [tf.clip_by_value(grad, -0.01, 0.01) for grad in enc_grads_])
        d_grads_ = enc_grads_clipped_ + dec_grads_
        d_vars_ = enc_vars_ + dec_vars_
        d_optim = d_opt.apply_gradients(zip(d_grads_, d_vars_))
    else:
        d_optim = d_opt.minimize(d_loss, var_list=enc_vars + dec_vars)
    g_optim = g_opt.minimize(g_loss, var_list=g_vars)

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/ae_loss", ae_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/cmd", cmd),
	tf.summary.scalar("loss/g_loss", g_loss),
	tf.summary.scalar("loss/d_loss", d_loss),
	tf.summary.scalar("misc/lr", lr),
    ])

    return (x, x_full, enc_x_full, z, z_full, x_test, x_precompute,
            x_test_precompute, avg_dist_x_to_x_test,
            avg_dist_x_to_x_test_precomputed, distances_xt_xp,
            mmd_to_cmd_indicator, g, g_full, ae_loss, d_loss, mmd, cmd, loss1,
            loss2, lr_update, d_optim, g_optim, summary_op)


def build_model_mmd_gan_simple(batch_size, gen_num, data_num, data_test_num,
        out_dim, z_dim, cmd_a, cmd_b):
    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_precompute')
    x_test_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_test_precompute')
    avg_dist_x_to_x_test_precomputed, distances_xt_xp = \
        avg_nearest_neighbor_distance(x_precompute, x_test_precompute)

    # Regular training placeholders.
    x = tf.placeholder(tf.float32, [batch_size, out_dim], name='x')
    z = tf.placeholder(tf.float32, [gen_num, z_dim], name='z')
    z_full = tf.placeholder(tf.float32, [data_num, z_dim], name='z_full')
    x_test = tf.placeholder(tf.float32, [batch_size, out_dim], name='x_test')
    avg_dist_x_to_x_test = tf.placeholder(tf.float32, shape=(),
        name='avg_dist_x_to_x_test')

    g, g_vars = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim)
    g_full, _ = generator(
        z_full, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)

    # SET UP MMD LOSS.
    mmd = compute_mmd(x, g, use_tf=True, slim_output=True)
    cmd = compute_cmd(x, g, k_moments=k_moments, use_tf=True, cmd_a=cmd_a,
        cmd_b=cmd_b)

    # Compare distances between data, heldouts, and gen.
    avg_dist_g_to_x, distances_g_x = avg_nearest_neighbor_distance(g, x)
    avg_dist_x_to_g, distances_x_g = avg_nearest_neighbor_distance(x, g)
    avg_dist_x_test_to_g, distances_xt_g = avg_nearest_neighbor_distance(x_test, g)
    loss1 = avg_dist_x_to_x_test - avg_dist_x_to_g
    loss2 = avg_dist_x_test_to_g - avg_dist_x_to_g

    g_loss = mmd

    lr = tf.Variable(learning_rate, name='lr', trainable=False)
    lr_update = tf.assign(lr, tf.maximum(lr * 0.8, 1e-8), name='lr_update')
    if optimizer == 'adagrad':
        g_opt = tf.train.AdagradOptimizer(lr)
    elif optimizer == 'adam':
        g_opt = tf.train.AdamOptimizer(lr)
    elif optimizer == 'rmsprop':
        g_opt = tf.train.RMSPropOptimizer(lr)
    else:
        g_opt = tf.train.GradientDescentOptimizer(lr)

    g_optim = g_opt.minimize(g_loss, var_list=g_vars)

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/g_loss", g_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/cmd", cmd),
	tf.summary.scalar("misc/lr", lr),
    ])

    return (x, z, z_full, x_test, x_precompute, x_test_precompute,
            avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
            distances_xt_xp, g, g_full, mmd, cmd, loss1, loss2, lr_update,
            g_optim, summary_op)


def build_model_kmmd_gan(batch_size, gen_num, data_num, data_test_num, out_dim,
        z_dim, cmd_a, cmd_b):
    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_precompute')
    x_test_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_test_precompute')
    avg_dist_x_to_x_test_precomputed, distances_xt_xp = \
        avg_nearest_neighbor_distance(x_precompute, x_test_precompute)

    # Regular training placeholders.
    x = tf.placeholder(tf.float32, [batch_size, out_dim], name='x')
    z = tf.placeholder(tf.float32, [gen_num, z_dim], name='z')
    z_full = tf.placeholder(tf.float32, [data_num, z_dim], name='z_full')
    x_test = tf.placeholder(tf.float32, [batch_size, out_dim], name='x_test')
    avg_dist_x_to_x_test = tf.placeholder(tf.float32, shape=(),
        name='avg_dist_x_to_x_test')

    g, g_vars = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim)
    g_full, _ = generator(
        z_full, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)

    # SET UP MMD LOSS.
    kmmd = compute_kmmd(x, g, k_moments=k_moments, kernel_choice=kernel_choice,
        use_tf=True, slim_output=True, sigma_list=[sigma])
    mmd = compute_mmd(x, g, use_tf=True, slim_output=True)
    cmd = compute_cmd(x, g, k_moments=k_moments, use_tf=True, cmd_a=cmd_a,
        cmd_b=cmd_b)

    # Compare distances between data, heldouts, and gen.
    avg_dist_g_to_x, distances_g_x = avg_nearest_neighbor_distance(g, x)
    avg_dist_x_to_g, distances_x_g = avg_nearest_neighbor_distance(x, g)
    avg_dist_x_test_to_g, distances_xt_g = avg_nearest_neighbor_distance(x_test, g)
    loss1 = avg_dist_x_to_x_test - avg_dist_x_to_g
    loss2 = avg_dist_x_test_to_g - avg_dist_x_to_g

    g_loss = kmmd

    lr = tf.Variable(learning_rate, name='lr', trainable=False)
    lr_update = tf.assign(lr, tf.maximum(lr * 0.8, 1e-8), name='lr_update')
    if optimizer == 'adagrad':
        g_opt = tf.train.AdagradOptimizer(lr)
    elif optimizer == 'adam':
        g_opt = tf.train.AdamOptimizer(lr)
    elif optimizer == 'rmsprop':
        g_opt = tf.train.RMSPropOptimizer(lr)
    else:
        g_opt = tf.train.GradientDescentOptimizer(lr)

    # Define optim nodes.
    # TODO: TEST CLIPPED GENERATOR.
    clip = 0
    if clip:
        g_grads_, g_vars_ = zip(*g_opt.compute_gradients(g_loss, var_list=g_vars))
        g_grads_clipped_ = tuple(
            [tf.clip_by_value(grad, -0.01, 0.01) for grad in g_grads_])
        g_optim = g_opt.apply_gradients(zip(g_grads_clipped_, g_vars_))
    else:
        g_optim = g_opt.minimize(g_loss, var_list=g_vars)

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/g_loss", g_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/kmmd", kmmd),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/cmd", cmd),
	tf.summary.scalar("misc/lr", lr),
    ])

    return (x, z, z_full, x_test, avg_dist_x_to_x_test, x_precompute,
            x_test_precompute, avg_dist_x_to_x_test_precomputed,
            distances_xt_xp, g, g_full, kmmd, mmd, cmd, loss1, loss2, g_optim,
            summary_op)


def build_model_cmd_gan(batch_size, gen_num, data_num, data_test_num, out_dim,
        z_dim, cmd_a, cmd_b):
    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_precompute')
    x_test_precompute = tf.placeholder(tf.float32, [data_test_num, out_dim],
        name='x_test_precompute')
    avg_dist_x_to_x_test_precomputed, distances_xt_xp = \
        avg_nearest_neighbor_distance(x_precompute, x_test_precompute)

    # Placeholders for regular training.
    x = tf.placeholder(tf.float32, [batch_size, out_dim], name='x')
    z = tf.placeholder(tf.float32, [gen_num, z_dim], name='z')
    z_full = tf.placeholder(tf.float32, [data_num, z_dim], name='z_full')
    x_test = tf.placeholder(tf.float32, [batch_size, out_dim], name='x_test')
    avg_dist_x_to_x_test = tf.placeholder(tf.float32, shape=(),
        name='avg_dist_x_to_x_test')
    prog_cmd_coefs = tf.placeholder(tf.float32, shape=(k_moments),
        name='prog_cmd_coefs')
    mmd_to_cmd_indicator = tf.placeholder(tf.float32, shape=(),
        name='mmd_to_cmd_indicator')

    g, g_vars = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim)
    g_full, _ = generator(
        z_full, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)

    # Compare distances between data, heldouts, and gen.
    avg_dist_g_to_x, distances_g_x = avg_nearest_neighbor_distance(g, x)
    avg_dist_x_to_g, distances_x_g = avg_nearest_neighbor_distance(x, g)
    avg_dist_x_test_to_g, distances_xt_g = avg_nearest_neighbor_distance(x_test, g)
    loss1 = avg_dist_x_to_x_test - avg_dist_x_to_g
    loss2 = avg_dist_x_test_to_g - avg_dist_x_to_g

    # SET UP CMD LOSS.
    # TODO: Experimental. Putting scale on moment exponent.
    cmd_k, cmd_k_terms = compute_cmd(x, g, k_moments=k_moments, use_tf=True,
        cmd_a=cmd_a, cmd_b=cmd_b, return_terms=True)  
    cmd_k_minus_k_plus_1 = compute_cmd(x, g, k_moments=k_moments+1, use_tf=True,
        cmd_a=cmd_a, cmd_b=cmd_b)
    mmd = compute_mmd(x, g, use_tf=True, slim_output=True)
    if cmd_variation == 'minus_k_plus_1':
        # With diverging k+1'th moment.
        cmd_adjusted = 2 * cmd_k - cmd_k_minus_k_plus_1
    elif cmd_variation == 'minus_mmd':
        # With diverging k* > k moments.
        cmd_adjusted = cmd_k - 0.01 * mmd
    elif cmd_variation == 'prog_cmd':
        # Progressively include higher moments.
        cmd_adjusted = tf.reduce_sum(prog_cmd_coefs * cmd_k_terms)
    elif cmd_variation == 'mmd_to_cmd':
        # Train with MMD, and switch to CMD at inidcator step.
        cmd_adjusted = (1 - mmd_to_cmd_indicator) * mmd + \
            (mmd_to_cmd_indicator) * cmd_k
    elif cmd_variation == 'fractional':
        # CMD with finer grid of moments up to k.
        cmd_adjusted = compute_cmd(x, g, k_moments=k_moments, use_tf=True,
            cmd_a=cmd_a, cmd_b=cmd_b, fractional_step=0.5)  
    else:
        cmd_adjusted = cmd_k  # Normal CMD loss.

    cmd = cmd_k
    g_loss = cmd_adjusted

    lr = tf.Variable(learning_rate, name='lr', trainable=False)
    lr_update = tf.assign(lr, tf.maximum(lr * 0.8, 1e-8), name='lr_update')
    if optimizer == 'adagrad':
        g_opt = tf.train.AdagradOptimizer(lr)
    elif optimizer == 'adam':
        g_opt = tf.train.AdamOptimizer(lr)
    elif optimizer == 'rmsprop':
        g_opt = tf.train.RMSPropOptimizer(lr)
    else:
        g_opt = tf.train.GradientDescentOptimizer(lr)

    # Define optim nodes.
    # TODO: TEST CLIPPED GENERATOR.
    clip = 0
    if clip:
        g_grads_, g_vars_ = zip(*g_opt.compute_gradients(g_loss, var_list=g_vars))
        g_grads_clipped_ = tuple(
            [tf.clip_by_value(grad, -0.01, 0.01) for grad in g_grads_])
        g_optim = g_opt.apply_gradients(zip(g_grads_clipped_, g_vars_))
    else:
        g_optim = g_opt.minimize(g_loss, var_list=g_vars)

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/g_loss", g_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/cmd", cmd),
	tf.summary.scalar("misc/lr", lr),
    ])

    return (x, z, z_full, x_test, x_precompute, x_test_precompute,
            avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
            distances_xt_xp, prog_cmd_coefs, mmd_to_cmd_indicator, cmd_k_terms,
            g, g_full, mmd, cmd, loss1, loss2, lr_update, g_optim, summary_op)


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
     data_raw_std) = load_data(data_num, percent_train)
    normed_moments_data = compute_moments(data, k_moments=k_moments+1)
    normed_moments_data_test = compute_moments(data_test, k_moments=k_moments+1)
    nmd_zero_indices = np.argwhere(np.array(normed_moments_data) < 0.1) 

    # Compute baseline statistics on moments for data set.
    num_samples = 100
    baseline_moments = np.zeros((num_samples, k_moments))
    for i in range(num_samples):
        d, _, d_num, _, _, d_raw_mean, d_raw_std = load_data(data_num,
            percent_train)
        d = d * d_raw_std + d_raw_mean
        baseline_moments[i] = compute_moments(d, k_moments)
    baseline_moments_means = np.mean(baseline_moments, axis=0)  # Mean.
    baseline_moments_stds = np.std(baseline_moments, axis=0)  # Standard deviation.
    baseline_moments_maes = np.mean(np.abs(
        baseline_moments - baseline_moments_means), axis=0)  # Mean absolute error.
    for j in range(k_moments):
        print('Moment {} Mean: {:.2f}, Std: {:.2f}, MAE: {:.2f}'.format(j+1,
            baseline_moments_means[j], baseline_moments_stds[j],
            baseline_moments_maes[j]))

    # Get compact interval bounds for CMD computations.
    cmd_a = np.min(data)
    cmd_b = np.max(data)
    print('cmd_span_const: {:.2f}'.format(1.0 / (np.abs(cmd_b - cmd_a))))
    
    # Prepare logging, checkpoint, and plotting directories.
    log_dir, checkpoint_dir, plot_dir, g_out_dir = prepare_dirs(load_existing)
    save_tag = str(args)
    with open(os.path.join(log_dir, 'save_tag.txt'), 'w') as save_tag_file:
        save_tag_file.write(save_tag)
    print('Save tag: {}'.format(save_tag))

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
    if model_type == 'ae_base':
        # Define compact space for CMD.
        (x, x_full, enc_x_full, x_test, x_precompute, x_test_precompute,
         avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
         distances_xt_xp, g, g_full, ae_loss, ae_base_loss,
         enc_norm_loss, d_loss, mmd, cmd, loss1, loss2, lr_update,
         d_optim, summary_op, h_weights, h_weights_update) = \
             build_model_ae_base(
                 batch_size, data_num, data_test_num, gen_num, out_dim, z_dim,
                 cmd_a, cmd_b)

    elif model_type == 'mmd_gan':
        (x, x_full, enc_x_full, z, z_full, x_test, x_precompute,
         x_test_precompute, avg_dist_x_to_x_test,
         avg_dist_x_to_x_test_precomputed, distances_xt_xp,
         mmd_to_cmd_indicator, g, g_full, ae_loss, d_loss, mmd, cmd, loss1,
         loss2, lr_update, d_optim, g_optim, summary_op) = \
             build_model_mmd_gan(
                 batch_size, gen_num, data_num, data_test_num, out_dim, z_dim,
                 cmd_a, cmd_b)

    elif model_type == 'mmd_gan_simple':
        (x, z, z_full, x_test, x_precompute, x_test_precompute,
         avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
         distances_xt_xp, g, g_full, mmd, cmd, loss1, loss2, lr_update, g_optim,
         summary_op) = \
             build_model_mmd_gan_simple(
                 batch_size, gen_num, data_num, data_test_num, out_dim, z_dim,
                 cmd_a, cmd_b)

    elif model_type == 'kmmd_gan':
        (x, z, z_full, x_test, avg_dist_x_to_x_test, x_precompute,
         x_test_precompute, avg_dist_x_to_x_test_precomputed,
         distances_xt_xp, g, g_full, kmmd, mmd, cmd, loss1, loss2, g_optim,
         summary_op) = \
             build_model_kmmd_gan(
                 batch_size, gen_num, data_num, data_test_num, out_dim, z_dim,
                 cmd_a, cmd_b)

    elif model_type == 'cmd_gan':
        (x, z, z_full, x_test, x_precompute, x_test_precompute,
         avg_dist_x_to_x_test, avg_dist_x_to_x_test_precomputed,
         distances_xt_xp, prog_cmd_coefs, mmd_to_cmd_indicator, cmd_k_terms, g,
         g_full, mmd, cmd, loss1, loss2, lr_update, g_optim, summary_op) = \
             build_model_cmd_gan(
                 batch_size, gen_num, data_num, data_test_num, out_dim, z_dim,
                 cmd_a, cmd_b)

    ###########################################################################
    # Start session.
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=sess_config) as sess:
        saver, summary_writer = prepare_logging(log_dir, checkpoint_dir, sess)
        sess.run(init_op)

        # load_existing_model().
        if load_existing:
            could_load, checkpoint_counter = load_checkpoint(
                saver, sess, checkpoint_dir)
            if could_load:
                load_step = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            load_step = 0

        # Once, compute average distance from heldout data to training data.
        avg_dist_x_to_x_test_precomputed_, _ = sess.run(
            [avg_dist_x_to_x_test_precomputed, distances_xt_xp],
            feed_dict=
                {x_precompute: data[:len(data_test)],
                 x_test_precompute: data_test})

        # Containers to hold empirical and relative errors of moments.
        empirical_moments_gens = np.zeros(
            ((max_step - 0) / log_step, k_moments+1))
        relative_error_of_moments = np.zeros(
            ((max_step - 0) / log_step, k_moments+1))
        reom = relative_error_of_moments



        #######################################################################
        # train()
        start_time = time()
        for step in range(load_step, max_step):
            # Set up inputs for all models.
            random_batch_data = np.array(
                [data[d] for d in np.random.choice(len(data), batch_size)])
            random_batch_data_test = np.array(
                [data_test[d] for d in np.random.choice(
                    len(data_test), batch_size)])
            random_batch_z = get_random_z(gen_num, z_dim)
            shared_feed_dict = {
                x: random_batch_data,
                x_test: random_batch_data_test,
                avg_dist_x_to_x_test: avg_dist_x_to_x_test_precomputed_}

            # Do an optimization step.
            if model_type == 'ae_base':
                if ae_variation == 'laplace':
                    # Added in noise.
                    sess.run([h_weights_update, d_optim], shared_feed_dict)
                else:
                    sess.run(d_optim, shared_feed_dict)

            elif model_type == 'mmd_gan':
                # Define schedule of switching from MMD to CMD.
                if mmd_variation == 'mmd_to_cmd':
                    if step < 50000:
                        indicator_ = 0.
                    else:
                        indicator_ = 1.
                else:
                    indicator_ = 0.  # Stay on MMD, never switch.
                shared_feed_dict.update(
                    {z: random_batch_z,
                     mmd_to_cmd_indicator: indicator_})
                sess.run([d_optim, g_optim], shared_feed_dict)

            elif model_type in ['mmd_gan_simple', 'kmmd_gan']:
                shared_feed_dict.update({z: random_batch_z})
                sess.run(g_optim, shared_feed_dict)

            elif model_type in ['cmd_gan']:
                # Compute coefficients for progressive CMD.
                # TODO: Figure out what schedule of progression is best.
                #   e.g. incremental 1, 2, ...; or perhaps evens then odds?
                # This fades in M2 over 5k iters, M3 over 10k iters, etc.
                phase_in_interval = 5000.
                coefs_ = np.ones(k_moments)
                for k in range(1, k_moments):
                    # Recall, by position index. Even positions are odd moments.
                    # positions: 0, 1, 2, 3, ...
                    # moments:   1, 2, 3, 4, ...
                    #if k % 2 == 0:
                    #    coefs_[k] = 0.0
                    #else:
                    #    pass
                    coefs_[k] = np.minimum(1., step / (k * phase_in_interval))

                # Define schedule of switching from MMD to CMD.
                if step < 50000:
                    indicator_ = 0.0
                else:
                    indicator_ = 1.0

                shared_feed_dict.update(
                    {z: random_batch_z,
                     prog_cmd_coefs: coefs_,
                     mmd_to_cmd_indicator: indicator_})

                sess.run(g_optim, shared_feed_dict)

            # Occasionally update learning rate.
            if step % lr_update_step == lr_update_step - 1:
                sess.run(lr_update)

            ###################################################################
            # log()
            # Occasionally log/plot results.
            if step % log_step == 0 and step > 0:
                # Read off from graph.
                if model_type == 'ae_base':
                    (ae_loss_, ae_base_loss_, enc_norm_loss_, d_loss_,
                     mmd_, g_batch_, summary_result, loss1_, loss2_) = \
                        sess.run(
                            [ae_loss, ae_base_loss, enc_norm_loss, d_loss,
                             mmd, g, summary_op, loss1, loss2],
                            shared_feed_dict)
                    print(('AE_BASE. Iter: {}, ae_loss: {:.4f}, '
                           'ae_base_loss: {:.4f}, enc_norm_loss: {:.4f}, '
                           'd_loss: {:.4f}, mmd: {:.4f}, loss1: {:.4f}, '
                           'loss2: {:.4f}').format(
                               step, ae_loss_, ae_base_loss_,
                               enc_norm_loss_, d_loss_, mmd_, loss1_,
                               loss2_))

                    shared_feed_dict.update({x_full: data})
                    h_weights_, enc_x_full_, g_full_normed_ = sess.run(
                        [h_weights, enc_x_full, g_full], shared_feed_dict)
                    # Extra troubleshooting logs.
                    #enc_x_full_ = sess.run(enc_x_full, shared_feed_dict)
                    #h_weights_, enc_x_full_ = sess.run([h_weights, enc_x_full],
                    #    shared_feed_dict)
                    print('NORM: {}'.format(np.sum(np.square(h_weights_))))

                elif model_type == 'mmd_gan':
                    d_loss_, ae_loss_, mmd_, loss1_, loss2_, summary_result = sess.run(
                        [d_loss, ae_loss, mmd, loss1, loss2, summary_op],
                        shared_feed_dict)
                    print(('MMD_GAN. Iter: {}, d_loss: {:.4f}, ae_loss: {:.4f}, '
                            'mmd: {:.4f}, loss1: {:.4f}, loss2: {:.4f}').format(
                               step, d_loss_, ae_loss_, mmd_, loss1_, loss2_))
                    enc_x_full_, g_full_normed_ = sess.run([enc_x_full, g_full],
                        {z_full: get_random_z(data_num, z_dim),
                         x_full: data})

                elif model_type == 'mmd_gan_simple':
                    mmd_, loss1_, loss2_, summary_result = sess.run(
                        [mmd, loss1, loss2, summary_op], shared_feed_dict)
                    g_full_normed_ = sess.run(g_full, feed_dict={
                        z_full: get_random_z(data_num, z_dim)})
                    print(('MMD_GAN_SIMPLE. Iter: {}, mmd: {:.4f}, loss1: {:.4f}, '
                           'loss2: {:.4f}').format(step, mmd_, loss1_, loss2_))

                elif model_type == 'kmmd_gan':
                    kmmd_, mmd_, loss1_, loss2_, summary_result = sess.run(
                        [kmmd, mmd, loss1, loss2, summary_op], shared_feed_dict)
                    g_full_normed_ = sess.run(g_full, feed_dict={
                        z_full: get_random_z(data_num, z_dim)})
                    print(('KMMD_GAN. Iter: {}, kmmd: {:.4f}, mmd: {:.4f}, '
                           'loss1: {:.4f}, loss2: {:.4f}').format(
                               step, kmmd_, mmd_, loss1_, loss2_))

                elif model_type == 'cmd_gan':
                    cmd_, cmd_k_terms_, mmd_, loss1_, loss2_, summary_result = \
                        sess.run(
                            [cmd, cmd_k_terms, mmd, loss1, loss2, summary_op],
                            shared_feed_dict)
                    g_full_normed_ = sess.run(g_full, feed_dict={
                        z_full: get_random_z(data_num, z_dim)})
                    print(('CMD_GAN. Iter: {}, cmd: {:.4f}, mmd: {:.4f} '
                           'loss1: {:.4f}, loss2: {:.4f}').format(
                               step, cmd_, mmd_, loss1_, loss2_))

                # TODO: DIAGNOSE NaNs.
                if np.isnan(mmd_):
                    pdb.set_trace()
                kmmd__ = compute_kmmd(data[:100], g_full_normed_[:100],
                    sigma_list=[sigma], k_moments=k_moments,
                    kernel_choice=kernel_choice, verbose=0)

                ###############################################################
                # Unormalize data and simulations for all logs and plots.
                g_full_unnormed = unnormalize(
                    g_full_normed_, data_raw_mean, data_raw_std)
                data_unnormed = unnormalize(
                    data, data_raw_mean, data_raw_std)
                data_test_unnormed = unnormalize(
                    data_test, data_raw_mean, data_raw_std)

                # Compute disclosure risk.
                (sensitivity, precision, false_positive_rate, tp, fn, fp,
                 tn) = evaluate_presence_risk(
                    data_unnormed, data_test_unnormed, g_full_unnormed,
                    )
                    #ball_radius=avg_dist_x_to_x_test_precomputed_)
                sens_minus_fpr = sensitivity - false_positive_rate
                print('  Sens={:.4f}, Prec={:.4f}, Fpr: {:.4f}, '
                      'tp: {}, fn: {}, fp: {}, tn: {}'.format(
                          sensitivity, precision, false_positive_rate, tp, fn,
                          fp, tn))

                # Add presence discloser stats to summaries.
                summary_writer.add_summary(summary_result, step)
                add_nongraph_summary_items(summary_writer, step,
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
                    print('  Time (s): {:.2f}, time/iter: {:.4f},'
                            ' Total est.: {:.4f}').format(
                                step, elapsed_time, time_per_iter, total_est_str)

                    print('  Save tag: {}'.format(save_tag))

                ################################################################
                # PLOT data and simulations.
                if out_dim == 1:
                    fig, ax = plt.subplots()
                    ax.hist(data_unnormed, normed=True, bins=30, color='gray', alpha=0.8,
                        label='data')
                    ax.hist(data_test_unnormed, normed=True, bins=30, color='red', alpha=0.3,
                        label='test')
                    ax.hist(g_full_unnormed, normed=True, bins=30, color='green', alpha=0.8,
                        label='gens')
                    plt.legend()
                    plt.savefig(os.path.join(plot_dir, '{}.png'.format(step)))
                    plt.close(fig)
                elif out_dim == 2:
                    fig, ax = plt.subplots()
                    ax.scatter(*zip(*data_unnormed), color='gray', alpha=0.2, label='data')
                    ax.scatter(*zip(*data_test_unnormed), color='red', alpha=0.2, label='test')
                    ax.scatter(*zip(*g_full_unnormed), color='green', alpha=0.2, label='sim')
                    ax.legend()
                    ax.set_title(tag)
                    plt.savefig(os.path.join(
                        plot_dir, '{}.png'.format(step)))
                    plt.close(fig)
                    
                # PLOT moment diagnostics.
                if data_dimension <= 2:
                    normed_moments_gens = compute_moments(g_full_normed_,
                        k_moments=k_moments+1)
                    empirical_moments_gens[step / log_step] = normed_moments_gens

                    ###########################################################
                    # Plot empirical moments throughout training.
                    fig, (ax_data, ax_gens) = plt.subplots(2, 1)
                    cmap = plt.cm.get_cmap('cool', k_moments+1)
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

                    ###########################################################
                    # Plot relative error of moments.
                    print('  data_normed moments: {}'.format(
                        normed_moments_data))
                    relative_error_of_moments_test = np.round(
                        (np.array(normed_moments_data_test) -
                         np.array(normed_moments_data)) /
                         np.array(normed_moments_data), 2)
                    relative_error_of_moments_gens = np.round(
                        (np.array(normed_moments_gens) -
                         np.array(normed_moments_data)) /
                         np.array(normed_moments_data), 2)

                    relative_error_of_moments_test[nmd_zero_indices] = 0.0
                    relative_error_of_moments_gens[nmd_zero_indices] = 0.0
                    reom[step / log_step] = relative_error_of_moments_gens

                    print('    RELATIVE_TEST: {}'.format(list(
                        relative_error_of_moments_test)))
                    print('    RELATIVE_GENS: {}'.format(list(
                        relative_error_of_moments_gens)))

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
                    print('  data_normed moments: {}'.format(
                        normed_moments_data))
                    print('  test_normed moments: {}'.format(
                        normed_moments_data_test))
                    print('  gens_normed moments: {}'.format(
                        normed_moments_gens))

                    ###########################################################
                    # Plot encoding space.
                    if model_type in ['ae_base', 'mmd_gan']:
                        if enc_x_full_.shape[1] == 2:
                            fig, ax = plt.subplots()
                            ax.scatter(*zip(*enc_x_full_), color='gray', alpha=0.2, label='enc')
                            ax.legend()
                            ax.set_title(tag)
                            plt.savefig(os.path.join(
                                plot_dir, 'enc_{}.png'.format(step)))
                            plt.close(fig)


if __name__ == "__main__":
    main()
