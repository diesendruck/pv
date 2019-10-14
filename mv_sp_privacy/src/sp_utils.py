import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
from scipy.spatial.distance import pdist
import sys
import tensorflow as tf
import time


from scipy.stats import multivariate_normal


def scale_01(data):
    """Scales data to fit in unit cube [0,1]^d."""
    data -= np.min(data, axis=0)  # Scales to [0, max]
    data /= np.max(data, axis=0)  # Scales to [0, 1]
    assert np.min(data) >= 0 and np.max(data) <= 1, 'Scaling incorrect.'
    return data

def get_support_points(x, num_support, max_iter=1000, lr=1e-2, is_tf=False,
                       power=None, y_init_option='random',
                       clip='bounds', do_wlb=False,
                       plot=True, do_mmd=False, mmd_sigma=None):
    """Initializes and gets support points.
    
    Args:
      x: Data. ND numpy array of any length, e.g. (100, dim).
      num_support: Scalar.
      max_iter: Scalar, number of times to loop through updates for all vars.
      lr: Scalar, amount to move point with each gradient update.
      is_tf: Boolean, chooses TensorFlow optimization.
      power: Int, power of energy metric.
      clip: Chooses how to clip SPs. [None, 'data', 'bounds'], where bounds 
        are [0,1].
      do_wlb: Boolean, chooses to use Exponential random weight for 
        each data point.
      do_mmd: Boolean, chooses MMD instead of energy distance.
      mmd_sigma: Float, bandwidth of MMD kernel.

    Returns:
      y_opt: (max_iter,N,D)-array. Trace of generated proposal points.
      e_opt: Float, energy between data and last iteration of y_out.
    """
    assert power >= 1, 'Power must be >= 1.'

        
    print('\nSTARTING RUN. is_tf: {}, y_init: {}, clip: {}, wlb: {}'.format(
        is_tf, y_init_option, clip, do_wlb))
    
    # Initialize generated particles for both sets (y and y_).
    d = x.shape[1]
    offset = 0.1 * (np.max(x) - np.min(x))

    if y_init_option == 'grid':
        grid_size = int(np.sqrt(num_support))
        assert grid_size == np.sqrt(num_support), \
            'num_support must square for grid'
        #_grid = np.linspace(offset, 1 - offset, grid_size)
        _grid = np.linspace(np.min(x) + offset, np.max(x) - offset, grid_size)
        y = np.array(np.meshgrid(_grid, _grid)).T.reshape(-1, d)
        # Perturb grid in order to give more diverse gradients.
        y += np.random.normal(0, 0.005, size=y.shape)

    elif y_init_option == 'random':
        #y = np.random.uniform(offset, 1 - offset, size=(num_support, d))
        offset = 0.1 * (np.max(x) - np.min(x))
        y = np.random.uniform(np.min(x) + offset, np.max(x) - offset,
                              size=(num_support, d))

    elif y_init_option == 'radial':
        positions = np.linspace(0,
                                (2 * np.pi) * num_support / (num_support + 1),
                                num_support)
        y = [[0.5 + 0.25 * np.cos(rad), 0.5 + 0.25 * np.sin(rad)] for
             rad in positions]
        y = np.array(y)
        
    elif y_init_option == 'subset':
        y = x[np.random.randint(len(x), size=num_support)] + \
            np.random.normal(0, 0.0001, size=[num_support, x.shape[1]])
        #y = np.array(x[np.random.randint(len(x), size=num_support)])
        
    elif y_init_option == 'uniform':
        y = np.random.uniform(np.min(x), np.max(x), size=[num_support, x.shape[1]])
        
    # For MMD, must supply sigma.
    if do_mmd:
        assert mmd_sigma is not None, 'If using MMD, must supply mmd_sigma.'

    # Optimize particles for each dataset (x0 and x1).
    y_opt, e_opt = optimize_support_points(x, y, max_iter=max_iter,
                                           learning_rate=lr, is_tf=is_tf,
                                           power=power,
                                           #save_iter=[int(max_iter / 2), max_iter - 1],  # PICK A SAVE_ITER.
                                           #save_iter=[5, 10, 50, 100, max_iter - 1],
                                           #save_iter=max_iter - 1,
                                           save_iter=100,
                                           clip=clip,
                                           do_wlb=do_wlb,
                                           do_mmd=do_mmd,
                                           mmd_sigma=mmd_sigma,
                                           plot=plot)

    # Get last updated set as support points.
    sp = y_opt[-1]

    return sp, e_opt


def optimize_support_points(data, gen, max_iter=500, learning_rate=1e-2,
                            is_tf=False, power=None, save_iter=100,
                            clip='bounds', do_wlb=False, do_mmd=False,
                            mmd_sigma=None, plot=True):
    """Runs TensorFlow optimization, n times through proposal points.
    
    Args:
      data: ND numpy array of any length, e.g. (100, dim).
      gen: ND numpy array of any length, e.g. (10, dim).
      max_iter: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      is_tf: Boolean, chooses TensorFlow optimization.
      power: Int, power of energy metric.
      clip: Chooses how to clip SPs. [None, 'data', 'bounds'], where bounds 
        are [0,1].
      do_wlb: Boolean, chooses to use Exponential random weight for 
        each data point.
      do_mmd: Boolean, chooses MMD instead of energy distance.
      mmd_sigma: Float, bandwidth of MMD kernel.
      plot: Boolean, to plot or not to plot.

    Returns:
      y_opt: (max_iter,N,D)-array. Trace of generated proposal points.
      e_opt: Float, energy between data and last iteration of y_out.
    """
    assert power >= 1, 'Power must be >= 1.'

    
    # Create WLB weights.
    if do_wlb:
        # Create (M,1) NumPy array with Exponential random weight for each data point.
        exp_weights = np.random.exponential(scale=1,
                                            size=(data.shape[0], 1)).astype(np.float32)
    else:
        exp_weights = None

    # Set up TensorFlow optimization.
    if is_tf:
        print('  [*] Using TensorFlow optimization.')
        
        # Set up container for results.
        y_out = np.zeros((max_iter, gen.shape[0], gen.shape[1]))
        
        # Build TensorFlow graph.
        tf.reset_default_graph()
        #tf_input_data = tf.placeholder(tf.float32, [data.shape[0], data.shape[1]],
        #                               name='input_data')
        tf_input_data = tf.placeholder(tf.float32, [None, data.shape[1]],
                                       name='input_data')
        
        tf_candidate_sp = tf.Variable(gen, name='sp', dtype=tf.float32)

        if do_mmd:
            tf_e_out, _ = mmd(tf_input_data, tf_candidate_sp, sigma=mmd_sigma,
                              is_tf=True, weights=exp_weights)
        else:
            tf_e_out, _ = energy(tf_input_data, tf_candidate_sp, power=power,
                                 is_tf=True, weights=exp_weights)

        opt = tf.train.AdamOptimizer(learning_rate)
        tf_grads, tf_variables = zip(*opt.compute_gradients(tf_e_out))
        tf_optim = opt.apply_gradients(zip(tf_grads, tf_variables))

        # Initialize graph.
        tf_init_op = tf.global_variables_initializer()
        tf_gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        tf_sess_config = tf.ConfigProto(allow_soft_placement=True,
                                        gpu_options=tf_gpu_options)

        # Run training.
        with tf.Session(config=tf_sess_config) as sess:
            sess.run(tf_init_op)

            start_time = time.time()

            for it in range(max_iter):
                max_before_batching = 256
                if len(data) <= max_before_batching:
                    # Do update over entire data set. [TAKES LONGER]
                    data_, sp_, e_, e_grads_, e_vars_ = sess.run(
                        [tf_input_data, tf_candidate_sp, tf_e_out, tf_grads,
                         tf_variables],
                        {tf_input_data: data})
                    sess.run([tf_optim], {tf_input_data: data})
                else:
                    batch_size = min(len(data), max_before_batching)
                    batch_data = data[np.random.choice(len(data),
                                                       batch_size,
                                                       replace=False)]             
                
                    data_, sp_, e_, e_grads_, e_vars_ = sess.run(
                        [tf_input_data, tf_candidate_sp, tf_e_out, tf_grads,
                         tf_variables],
                        {tf_input_data: batch_data})
                    sess.run([tf_optim], {tf_input_data: batch_data})

                # TODO: Decide whether to clip support points to domain bounds.
                if clip == 'bounds':
                    sp_ = np.clip(sp_, 0, 1)
                elif clip == 'data':
                    sp_ = np.clip(sp_, np.min(data), np.max(data))

                    
                # Store result in container.
                y_out[it, :] = sp_

                # Plot occasionally.
                if (plot == True and
                    data.shape[1] == 2 and
                    it % save_iter == 0 and
                    #it in save_iter and
                    it > 0
                   ):
                    if it > 0:
                        print('  [*] Overall it/s: {:.4f}'.format(
                            (time.time() - start_time) / it))

                    plt.scatter(data[:, 0], data[:, 1], c='gray', s=64,
                                alpha=0.3, label='data')
                    plt.scatter(sp_[:, 0], sp_[:, 1], s=32, c='limegreen',
                                label='sp')

                    # Plot arrows of gradients at each point.
                    #for i, sp in enumerate(sp_):
                    #    plt.arrow(sp[0], sp[1], *(-1. * e_grads_[0][i]),
                    #              color='white', head_width=0.02,
                    #              head_length=0.02, length_includes_head=False)

                    plt.title('it: {}, e_out: {:.8f}'.format(it, e_))
                    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    #plt.xlim(0, 1)
                    #plt.ylim(0, 1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.show()
                    
                elif (plot == True and
                      it % save_iter == 0 and
                      #it in save_iter and
                      it > 0
                     ):
                    graph = pd.plotting.scatter_matrix(pd.DataFrame(sp_), figsize=(10,10))
                    plt.suptitle('SP Optimization. num_supp={}, it={}, e={:.6f}'.format(len(sp_), it, e_))
                    plt.show()
                    
            print('  [*] Time elapsed: {:.2f}'.format(time.time() - start_time))
    
    
    # Set up NumPy optimization.
    elif not is_tf:
        print('  [*] Using analytical gradient optimization.')
        
        # Set up container for results.
        y_out = np.zeros((max_iter, gen.shape[0], gen.shape[1]))

        # Run optimization steps.
        for it in range(max_iter):
            
            # Compute distance and return value and gradients.
            if do_mmd:
                e_, e_grads = mmd(data, gen, sigma=mmd_sigma)
                gen -= learning_rate * e_grads
            else:
                e_, e_grads = energy(data, gen, power=power)
                gen -= learning_rate * e_grads

            # -----------------------------------------------------------------
            # TODO: Decide whether to clip support points to domain bounds.
            if clip == 'bounds':
                gen = np.clip(gen, 0, 1)
            elif clip == 'data':
                gen = np.clip(gen, np.min(data), np.max(data))
            # -----------------------------------------------------------------

            y_out[it, :] = gen

            # Plot occasionally.
            if (data.shape[1] == 2 and
                it in save_iter and
                it > 0 and
                plot == True
               ):
                plt.scatter(data[:, 0], data[:, 1], c='gray', s=64, alpha=0.3,
                            label='data')
                plt.scatter(gen[:, 0], gen[:, 1], s=32, c='limegreen',
                            label='sp')

                # Plot arrows of gradients at each point.
                #for i, sp in enumerate(gen):
                #    plt.arrow(sp[0], sp[1], *(-1. * e_grads[i]),
                #              color='white', head_width=0.02,
                #              head_length=0.02, length_includes_head=False)

                plt.title('it: {}, e_out: {:.8f}'.format(it, e_))
                #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                #plt.xlim(0, 1)
                #plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.show()
    

    return y_out, e_


def energy(data, gen, power=None, is_tf=False, weights=None, return_k=False):
    """Computes abbreviated energy statistic between two point sets.

    The smaller the value, the closer the sets.
    
    Args:
      data: ND NumPy array of any length, e.g. (1000, 2).
      gen: ND NumPy array of any length, e.g. (10, 2).
      power: Exponent in distance metric. Must be >= 1.
      is_tf: Boolean. Selects for TensorFlow functions.
      weights: (M,1) NumPy array with random weight for each data point.
      return_k: Boolean, whether to return energy matrix.
    
    Returns:
      e: Scalar, the energy between the sets.
      gradients_e: Numpy array of energy gradients for each proposal point.
      K: Energy matrix.
    """
    assert power >= 1, 'Power must be >= 1.'

    # ------------- TensorFlow VERSION -------------

    if is_tf:
        x = data
        y = gen
        dim = tf.shape(x)[1]
        data_num = tf.shape(x)[0]
        gen_num = tf.shape(y)[0]

        # Compute energy.
        v = tf.concat([x, y], 0)
        v_tiled_across = tf.tile(tf.expand_dims(v, 1), [1, tf.shape(v)[0], 1])
        v_tiled_down = tf.tile(tf.expand_dims(v, 0), [tf.shape(v)[0], 1, 1])
        pairwise_difs = v_tiled_across - v_tiled_down

        # TODO: WHY IS THIS NEEDED TO AVOID NANs?
        # Replace diagonals (which are zeros) with small value.
        # Compute K (norm of pairwise differences) with filler on diagonal,
        #   then zero out diagonal.
        diag_filler = 1e-10 * tf.eye(tf.shape(pairwise_difs)[0])
        diag_filler = tf.tile(
            tf.expand_dims(diag_filler, 2),
            [1, 1, tf.shape(pairwise_difs)[2]])
        pairwise_difs = pairwise_difs + diag_filler

        # Build kernel matrix using norms of pairwise difs.
        
        
        # Build kernel matrix, and optionally multiple by data weights.
        K = tf.norm(pairwise_difs, axis=2, ord=power)
        #K = tf.pow(tf.norm(pairwise_difs, axis=2, ord='euclidean'), power)  # TODO: testing new norm strategy
        K = tf.matrix_set_diag(K, tf.zeros(tf.shape(v)[0]))  # Zero-out diagonal.
        if weights is not None:
            weights = tf.constant(weights)
            p1_gen_num_weights = tf.tile(weights, (1, gen_num))
            K_xy = K[:data_num, data_num:] * p1_gen_num_weights
        else:
            K_xy = K[:data_num, data_num:]
        K_xx = K[:data_num, :data_num]
        K_yy = K[data_num:, data_num:]

        m = tf.cast(data_num, tf.float32)
        n = tf.cast(gen_num, tf.float32)

        e = (2. / n / m * tf.reduce_sum(K_xy) -
             1. / m / m * tf.reduce_sum(K_xx) -
             1. / n / n * tf.reduce_sum(K_yy))
        
        #e = tf.sqrt(e)  # TODO: Testing.

        gradients_e = None
    
    
    # ------------- NumPy VERSION -------------

    else:
        x = data
        y = gen
        dim = x.shape[1]
        data_num = len(x)
        gen_num = len(y)
        
        # Compute energy.
        v = np.concatenate((x, y), 0)
        v_tiled_across = np.tile(v[:, np.newaxis, :], (1, v.shape[0], 1))
        v_tiled_down = np.tile(v[np.newaxis, :, :], (v.shape[0], 1, 1))
        pairwise_difs = v_tiled_across - v_tiled_down

        # Build kernel matrix, and optionally multiple by data weights.
        K = np.linalg.norm(pairwise_difs, axis=2, ord=power)
        if weights is not None:
            p1_gen_num_weights = np.tile(weights, (1, gen_num))
            K_xy = K[:data_num, data_num:] * p1_gen_num_weights
        else:
            K_xy = K[:data_num, data_num:]
        K_xx = K[:data_num, :data_num]
        K_yy = K[data_num:, data_num:]

        e = (2. / gen_num / data_num * np.sum(K_xy) -
             1. / data_num / data_num * np.sum(K_xx) -
             1. / gen_num / gen_num * np.sum(K_yy))
        
        #e = np.sqrt(e)  # TODO: Testing.
        
        
        # TODO: COMPUTE GRADIENTS FOR WEIGHTED DATA.
        if weights is not None:
            print('[*] Analytical gradients for weighted data not yet implemented')
            sys.exit()
        
        # Note: Term2 assumes y in first position. For y in second position,
        #       need to multiply grad_matrix by -1.
        if power == 1:
            term1 = np.sign(np.sum(pairwise_difs, axis=2))
            term2 = np.sign(pairwise_difs)
        else:
            c1 = 1. / power

            # Define term containing Infs on diag, then replace Infs w/ zero.
            term1 = c1 * np.sum(pairwise_difs ** power, axis=2) ** (c1 - 1)
            # np.fill_diagonal(term1, 0)
            term1[np.where(term1 == np.inf)] = 0

            term2 = power * (pairwise_difs ** (power - 1))

        assert len(term1[:, :, np.newaxis].shape) == len(term2.shape)
        grad_matrix = term1[:, :, np.newaxis] * term2
        grad_matrix_yx = grad_matrix[data_num:, :data_num]
        grad_matrix_yy = grad_matrix[data_num:, data_num:]

        gradients_e = np.zeros((gen_num, dim))
        for i in range(gen_num):
            grad_yi = (2. / data_num / gen_num * np.sum(grad_matrix_yx[i],
                                                        axis=0) -
                       2. / gen_num / gen_num * np.sum(grad_matrix_yy[i],
                                                       axis=0))
            gradients_e[i] = grad_yi


    if return_k:
        return e, gradients_e, K
    else:
        return e, gradients_e


def mmd(data, gen, sigma=1., is_tf=False, weights=None):
    """Computes MMD between NumPy arrays.

    The smaller the value, the closer the sets.
    
    Args:
      data: ND NumPy array of any length, e.g. (1000, 2).
      gen: ND NumPy array of any length, e.g. (10, 2).
      sigma: Float, kernel bandwidth.
      is_tf: Boolean. Selects for TensorFlow functions.
      weights: (M,1) NumPy array with random weight for each data point.
    
    Returns:
      mmd: Scalar, the MMD between the sets.
      gradients_mmd: NumPy array of MMD gradients for each generated point.
    """
    
    #print('  [*] Analytical gradients not yet implemented for MMD.')
    
    x = data
    y = gen
        
    # ------------- TensorFlow VERSION -------------

    if is_tf:
        dim = tf.shape(x)[1]
        data_num = tf.shape(x)[0]
        gen_num = tf.shape(y)[0]
        
        v = tf.concat([x, y], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        v_sq = tf.reshape(tf.diag_part(VVT), [-1, 1])
        
        #v_sq_tiled = tf.tile(v_sq, [1, v_sq.get_shape().as_list()[0]])
        #v_sq_tiled_T = tf.transpose(v_sq_tiled)
        v_sq_tiled = tf.tile(v_sq, [1, data_num + gen_num])
        v_sq_tiled_T = tf.transpose(v_sq_tiled)
        
        #v_sq_tiled = tf.tile(tf.expand_dims(v_sq, 1), [1, tf.shape(v_sq)[0], 1])
        #v_sq_tiled_T = tf.transpose(v_sq_tiled, [1, 0, 2])
        

        
        # Build kernel matrix, and optionally multiple by data weights.
        exp_object = v_sq_tiled - 2 * VVT + v_sq_tiled_T
        
        gamma = 1.0 / (2.0 * sigma**2)
        K = tf.exp(-gamma * exp_object)
        if weights is not None:
            weights = tf.constant(weights)
            p1_gen_num_weights = tf.tile(weights, (1, gen_num))
            K_xy = K[:data_num, data_num:] * p1_gen_num_weights            
        else:
            K_xy = K[:data_num, data_num:]
        K_xx = K[:data_num, :data_num]
        K_yy = K[data_num:, data_num:]
            
        m = tf.cast(data_num, tf.float32)
        n = tf.cast(gen_num, tf.float32)

        mmd = (1. / m / m * tf.reduce_sum(K_xx) +
               1. / n / n * tf.reduce_sum(K_yy) -
               2. / m / n * tf.reduce_sum(K_xy))
        
        # TODO: MMD gradients.
        gradients_mmd = None
        
        return mmd, gradients_mmd

    
    # ------------- NumPy VERSION -------------

    elif not is_tf:
        data_num = len(x)
        gen_num = len(y)

        if len(x.shape) == 1:
            x = np.reshape(x, [-1, 1])
            y = np.reshape(y, [-1, 1])
        v = np.concatenate((x, y), 0)
        VVT = np.matmul(v, np.transpose(v))
        sqs = np.reshape(np.diag(VVT), [-1, 1])
        sqs_tiled_horiz = np.tile(sqs, np.transpose(sqs).shape)
        
        # Build kernel matrix, and optionally multiple by data weights.
        exp_object = sqs_tiled_horiz - 2 * VVT + np.transpose(sqs_tiled_horiz)
        gamma = 1.0 / (2.0 * sigma**2)
        K = np.exp(-gamma * exp_object)
        if weights is not None:
            p1_gen_num_weights = np.tile(weights, (1, gen_num))
            K_xy = K[:data_num, data_num:] * p1_gen_num_weights
        else:
            K_xy = K[:data_num, data_num:]
        K_xx = K[:data_num, :data_num]
        K_yy = K[data_num:, data_num:]

        mmd = (1. / data_num / data_num * np.sum(K_xx) +
               1. / gen_num / gen_num * np.sum(K_yy) -
               2. / data_num / gen_num * np.sum(K_xy))

        # TODO: MMD gradients.
        gradients_mmd = None
        
        return mmd, gradients_mmd


def get_energy_sensitivity(data, reference_size, power=None):
    """Computes energy sensitivity.

    Args:
        data (array): Data set, from which dimension will be computed.
        reference_size (int): Number of points in reference set (either
          data size, or SP size if comparing to SP.

    Returns:
        energy_sensitivity (float): Sensitivity value.
    """
    assert power is not None, 'power is None. Define it.'
    dim = data.shape[1]

    # Define energy sensitivity for Exponential Mechanism.
    energy_sensitivity = (
        2 * dim ** (1. / power) * (2 * reference_size - 1) / reference_size ** 2)
    
    return energy_sensitivity


def plot_nd(d, w=10, h=10, title=None):
    graph = pd.plotting.scatter_matrix(pd.DataFrame(d), figsize=(w, h));
    if title:
        plt.suptitle(title)
    #plt.show()


def sample_sp_exp_mech(e_opt, energy_sensitivity, x, y_opt, method='mh', num_y_tildes=1,
                       alpha=None, plot=False, do_mmd=False,
                       mmd_sigma=None, burnin=5000, thinning=2000, partial_update=True,
                       save_dir=None, power=None, set_seed=False):
    """Samples in space of support points.

    Args:
      e_opt: Energy distance between optimal support points and data.
      energy_sensitivity: Sensitivity, based on user-selected privacy budget.
      x: NumPy array, data.
      y_opt: NumPy array, optimal support points.
      method: String, indicates which sampling method to use.
      num_y_tildes: Samples of support points to draw.
      alpha: Float, privacy budget.
      plot: Boolean, controls plotting.
      do_mmd: Boolean, chooses MMD instead of energy distance.
      mmd_sigma: Float, bandwidth of MMD kernel.
      burnin: Int, number of burned iterations in Metropolis Hastings.
      thinning: Int, thinning gap in Metropolis Hastings.
      partial_update: Boolean, in MH do random walk only on a random subset.
      save_dir: String, location to save plots.
      power: Int, power in energy metric.
      set_seed: Boolean, whether to set seed before sampling.

    Returns:
      y_tildes: NumPy array, sampled support point sets.
      energies: NumPy array, energies associated with sampled sets.
      energy_estimation_errors: NumPy array, relative error of energy approximation.
    """
    if plot == True:
        assert save_dir is not None, 'To plot, must define save_dir.'
    
    if do_mmd:
        print('[*] Using MMD as distribution distance.')
    
    sensitivity_string = ('\nPr(e) = a / (2U) * exp(- a / (2U) * e) '
                          '~ Exp(2U/a) = Exp(2 * {:.4f} / {:.3f}) = '
                          'Exp({:.8f})\n'.format(energy_sensitivity,
                                                 alpha,
                                                 2 * energy_sensitivity / alpha))
    print(sensitivity_string)
        
        
    # --------------------------------------------------------------
    # Sample support points.     
    # Let Y* be optimal support points. Initialize Y_t as Y*. Let
    # Y_t' = Y_t + random walk noise. For differential privacy level a
    # and sensitivity U, let acceptance ratio of Y_t' be
    #   \gamma = exp(a / (2U) * [e(Y_t, Y*) - e(Y_t', Y*)]).
    # Then accept Y_t' with probability min(1, \gamma).

    # Choose setup for Metropolis-Hastings.
    max_step_size = 0.25 * (np.max(x) - np.min(x))
    step_size = 1e-2
    chain_length = burnin + thinning * num_y_tildes
    print('Running chain. Length={}, Burn={}, Thin={}'.format(
        chain_length, burnin, thinning))

    # Initialize the support points Y_t.
    y_t = np.random.uniform(size=y_opt.shape)
    #y_t = y_opt

    # Choose distance metric, and compute initial value.
    if do_mmd:
        energy_t, _ = mmd(y_t, y_opt, sigma=mmd_sigma)
    else:
        energy_t, _ = energy(y_t, y_opt, power=power)

        
    # Create containers for markov chain results.
    y_mh = np.zeros(shape=(chain_length, y_opt.shape[0], y_opt.shape[1]))
    ratios_unthinned = np.zeros(chain_length)
    acceptance_rates = np.zeros(chain_length)
    energies_unthinned = np.zeros(chain_length)
    accepts = []

    # Factor for difference of energies in MH step.
    diff_factor = alpha / (2. * energy_sensitivity)
    print('Difference factor: {:.2f}\n'.format(diff_factor))

    # Store last 100 acceptance rates.
    trail_len = 50
    acceptance_rates_trailing = np.zeros(trail_len)
    
    
    if set_seed:
        np.random.seed(234)
    
    # Run chain.
    for i in range(chain_length):
        
        if partial_update:
            # Perturb one point at a time.
            update_i = i % len(y_t)
            y_t_candidate = np.copy(y_t)
            y_t_candidate[update_i] += np.random.normal(scale=step_size, size=y_t.shape[1])
        else:
            # Add random walk noise to current set of support points.
            y_update = np.random.normal(scale=step_size, size=y_t.shape)
            y_t_candidate = y_t + y_update

        # Clip candidate values to data domain of [0,1].
        y_t_candidate = np.clip(y_t_candidate, 0, 1)


        # Compute energy difference due to single-point change in candidate.
        """
        K_cc = K[:len(y_t), :len(y_t)]  # candidate-candidate term
        K_co = K[:len(y_t), len(y_t):]  # candidate-optimal term
        curr_co_i_row = K_co[update_i]
        curr_cc_i_row = K_cc[update_i]
        curr_cc_i_col = K_cc[:, update_i]
        new_co_i_row = np.linalg.norm(y_t_candidate[update_i] - y_opt, ord=power, axis=1)
        new_cc_i_row = np.linalg.norm(y_t_candidate[update_i] - y_t_candidate, ord=power, axis=1)
        new_cc_i_col = np.linalg.norm(y_t_candidate - y_t_candidate[update_i], ord=power, axis=1)
        e_diff = (
            2. / (len(y_t) ** 2) * np.sum(curr_co_i_row - new_co_i_row) -
            2. / (len(y_t) ** 2) * np.sum(curr_cc_i_row - new_cc_i_row))
        K[update_i, len(y_t):] = new_co_i_row
        K[update_i, :len(y_t)] = new_cc_i_row
        K[:len(y_t), update_i] = new_cc_i_col
        #... another update for oc.
        """
        
        K_cc_old = [np.linalg.norm(y_t[update_i] - y_t[i], ord=power) for i in range(len(y_t))]
        K_co_old = [np.linalg.norm(y_t[update_i] - y_opt[i], ord=power) for i in range(len(y_opt))]
        
        K_cc_new = [np.linalg.norm(y_t_candidate[update_i] - y_t_candidate[i], ord=power) for i in range(len(y_t))]
        K_co_new = [np.linalg.norm(y_t_candidate[update_i] - y_opt[i], ord=power) for i in range(len(y_opt))]
        
        part_e_old = (2. / (len(y_opt) ** 2)) * (np.sum(K_co_old) - np.sum(K_cc_old))
        part_e_new = (2. / (len(y_opt) ** 2)) * (np.sum(K_co_new) - np.sum(K_cc_new))
        
        
        # Compute the acceptance ratio.
        # With U = 2 * DIM ** (1. / POWER) * (2 * N - 1) / N ** 2
        # 
        #         exp(- a / (2U) * e_t')
        # ratio = ---------------------- = exp( a / (2U) * (e_t - e_t'))
        #         exp(- a / (2U) * e_t)
        #
        # As N increases, U decreases, and energy difference is magnified.
        # Also, as alpha increases, energy difference is magnified.
        # Therefore, with more support points and higher alpha, energy
        #   difference is magnified.
        
        # Compute metric for current and candidate.
        if do_mmd:
            energy_t, _ = mmd(y_t, y_opt, sigma=mmd_sigma)
            energy_t_candidate, _ = mmd(y_t_candidate, y_opt, sigma=mmd_sigma)
            energy_diff = energy_t_candidate - energy_t
        else:
            #energy_t, _ = energy(y_t, y_opt, power=power)
            #energy_t_candidate, _ = energy(y_t_candidate, y_opt, power=power)
            #energy_diff = energy_t_candidate - energy_t

            energy_diff = part_e_new - part_e_old
            energy_t_candidate = energy_t + energy_diff
            #assert np.isclose(energy_t_candidate, energy(y_t_candidate, y_opt, power=power)[0])


        ratios_unthinned[i] = np.exp(diff_factor * -1. * energy_diff)

        # print('e_t - e_t\' = {:.5f}, ratio = {:6f}'.format(
        #       energy_t - energy_t_candidate, ratios_unthinned[i]))
        # pdb.set_trace()

        # Accept or reject the candidate.
        if np.random.uniform() < ratios_unthinned[i]:    # Accept.
            accepts.append(i)
            y_t = y_t_candidate
            energy_t = energy_t_candidate
            y_mh[i] = y_t_candidate
            energies_unthinned[i] = energy_t_candidate
        else:                                            # Reject.
            y_mh[i] = y_t
            energies_unthinned[i] = energy_t


        # Adapt step size to keep acceptance rate around 30%.
        acceptance_rate = float(len(accepts)) / (i + 1)
        acceptance_rates[i] = acceptance_rate
        acceptance_rates_trailing[i % trail_len] = acceptance_rate
        if i % trail_len == trail_len - 1:
            avg = np.mean(acceptance_rates_trailing)
            #print('                      trail avg: {:.6f}'.format(avg))
            #print('s_s: {:.6f}'.format(step_size))

            if avg > 0.4:
                step_size *= 2
            elif avg > 0.3 and avg < 0.4:
                step_size *= 1.2
            elif avg > 0.2 and avg < 0.3:
                step_size *= 0.8
            elif avg < 0.2:
                step_size *= 0.5
            step_size = np.clip(step_size, 1e-5, max_step_size)
            
        if i % int(chain_length / 10) == 0:
        #if i == chain_length - 1:
            print('acceptance_rate={:.8f}, step_size={:.8f}'.format(acceptance_rate, step_size))
            print('Energy diff: {:.8f}'.format(energy_diff))


        # Plot the points.
        if plot and x.shape[1] == 2 and i % int(chain_length / 10) == 0:
            plt.scatter(x[:, 0], x[:, 1], c='gray', alpha=0.3,
                        label='data')
            plt.scatter(y_opt[:, 0], y_opt[:, 1], c='limegreen',
                        label='sp(data)')
            plt.scatter(y_mh[i][:, 0], y_mh[i][:, 1], c='red', alpha=1,
                        label='~sp(data)', marker='+')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'priv_sp_sample.png'))
            plt.show()

        elif plot and x.shape[1] > 2 and i % int(chain_length / 10) == 0:
            plot_nd(x, title='data')
            plot_nd(y_mh[i], title='MH ~SP. It: {}, e={:.6f}'.format(i, energies_unthinned[i]))


    # Thinned results.
    y_tildes = y_mh[burnin::thinning]
    ratios = ratios_unthinned[burnin::thinning]
    energies = energies_unthinned[burnin::thinning]
    # dups = [y_tildes[i] == y_tildes[i-1] for i in range(1, len(y_tildes))]

    energy_estimation_errors = None

    # Plot results of markov chain.
    if plot:
    #if 1:
        # Plot acceptance ratios.
        plt.plot(acceptance_rates)
        #plt.title('accept_ratios, median={:.5f}'.format(np.median(ratios)))
        plt.xlabel('Sample', fontsize=14)
        plt.ylabel('Acceptance rates', fontsize=14)
        plt.ylim((0,1))
        plt.tight_layout()
        #plt.savefig(os.path.join(save_dir, 'priv_sp_acceptance_ratios.png'))
        plt.show()

        # Plot traceplot of energies.
        plt.plot(energies_unthinned)
        #plt.title('energies, median={:.5f}'.format(np.median(energies)))
        plt.xlabel('Sample', fontsize=14)
        plt.ylabel(r'Energy, $e(y, \tilde{y})$', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'priv_sp_traceplot_energies.png'))
        plt.show()

        # Inspect correlation of energies.
        print('Acceptance rate: {:.3f}'.format(len(accepts) / chain_length))
        print('percent steps that improved energy score: {:.3f}'.format(
            sum(ratios_unthinned > 1.) / len(ratios_unthinned)))
        #plt.acorr(energies, maxlags=20)
        #plt.show()

        # Inspect distribution of energies.
        plt.title('Energies with MH, n={}'.format(len(energies)))
        plt.hist(energies, bins=20, alpha=0.3)
        plt.show()

    # --------------------------------------------------------------

    return y_tildes, energies, energy_estimation_errors


def sample_sp_exp_mech_gridwalk(energy_sensitivity, y_opt, x, alpha=None, power=None):
    """Samples in space of support points.

    For data points in R^d, energy power p, and num support points N, energy
    sensitivity U = 2 * d^(1 / p) * (2N - 1) / N^2. Grid-Walk samples proportional
    to F(y_t), where:

    F(y_t) = exp(-a / (2U) * e(y_t, y_opt)).

    Proposals during sampling are coordinate-wise, e.g. 10, 2-dim points make a
    20-dimensional input, where one component is perturbed at a time.

    L is the Lipschitz constant for f(y_t), where:

    f(y_t) = ln(F(y_t) = -a / (2U) * e(y_t, y_opt)

    and

    |f(y) - f(y')| <= L * (max_i |y_i - y'_i|)

    Changing a single component changes e() by a max of 2 * (2N - 1) / N^2.
    Combining with the rest of f(), f() changes by a max of
    -a / (2U) * 2 * (2N - 1) / N^2. Substituting in the value of U, f() changes
    by a max absolute amount L, where:

    L = a / (2 * d^(1 / p)).

    Cube size in Grid-Walk is 1 / ceil(L). For a=1000, d=2, p=2,

    1 / ceil(L) = 1 / ceil(1000 / (2 * 2^(1 / 2))) = 1 / 354.


    Args:
      energy_sensitivity: Sensitivity, based on user-selected privacy budget.
      y_opt: NumPy array, optimal support points.
      x: NumPy array, data.
      alpha: Float, privacy budget.
      power: Int, power in energy metric.

    Returns:
      y_tilde: NumPy array, sampled support point set.
    """

    # Define L.
    dim = y_opt.shape[1]
    L = alpha / (2 * dim ** (1 / power))
    cube_size = 1. / np.ceil(L)
    grid = np.linspace(cube_size / 2, 1 - cube_size / 2, num=1 / cube_size)
    
    # Choose random starting point. dim random coordinates for each of 
    num_supp = y_opt.shape[0]
    gw_dim = num_supp * dim
    y_t = np.array(
        [np.random.choice(grid) for _ in range(gw_dim)]).reshape(num_supp, dim)
    energy_t, _ = energy(y_t, y_opt, power=power)
    
    # Compute number of steps to take
    beta = 0.
    dp_delta = 1. / num_supp
    q_x0 = 1. / (len(grid) ** gw_dim)
    num_steps = np.int(np.ceil(
        16 * np.exp(4) * (gw_dim ** 2) * (L ** 2) * np.exp(2 * beta) *
        (np.log(np.exp(2) / dp_delta) + np.log(1. / q_x0))))
    
    # Helper for energy updates.
    diff_factor = alpha / (2. * energy_sensitivity)

    
    # Run the sampler.
    for i in range(num_steps):
        
        # With probability 1/2, do not move.
        if np.random.uniform() < 0.5:
            continue

        # Otherwise, propose move to adjacent block, and move if valid.
        # Randomly select which component to perturb.
        x_ind = np.random.choice(range(num_supp))
        y_ind = np.random.choice(range(dim))
        direction = np.random.choice([-1, 1])
        new_val = y_t[x_ind][y_ind] + direction * cube_size
        if new_val < 0. or new_val > 1.:
            continue
        
        # Copy y_t and apply new value to candidate.
        y_t_candidate = np.copy(y_t)
        y_t_candidate[x_ind][y_ind] = new_val

        # Compute the acceptance ratio,
        #          F(y_t_candidate)     exp(- a / (2U) * e_t')
        # ratio = ------------------ = ------------------------ = exp( a / (2U) * (e_t - e_t'))
        #          F(y_t)               exp(- a / (2U) * e_t)
        # using difference in energy. Since only one point changes, only compute
        # that part of the Gram matrix "K". Below, "cc" indicates the portion of K
        # between candidate and candidate, and "co" ... between candidate and optimal.
        update_ind = x_ind
        K_cc_old = [np.linalg.norm(y_t[update_ind] - y_t[ind], ord=power) for ind in range(len(y_t))]
        K_co_old = [np.linalg.norm(y_t[update_ind] - y_opt[ind], ord=power) for ind in range(len(y_opt))]
        
        K_cc_new = [np.linalg.norm(y_t_candidate[update_ind] - y_t_candidate[ind], ord=power) for ind in range(len(y_t))]
        K_co_new = [np.linalg.norm(y_t_candidate[update_ind] - y_opt[ind], ord=power) for ind in range(len(y_opt))]
        
        part_e_old = (2. / (len(y_opt) ** 2)) * (np.sum(K_co_old) - np.sum(K_cc_old))
        part_e_new = (2. / (len(y_opt) ** 2)) * (np.sum(K_co_new) - np.sum(K_cc_new))
        
        energy_diff = part_e_new - part_e_old
        energy_t_candidate = energy_t + energy_diff
        ratio = np.exp(diff_factor * -1. * energy_diff)

        # Accept or reject the candidate.
        if np.random.uniform() < ratio:    # Accept.
            y_t = y_t_candidate
            energy_t = energy_t_candidate
        # Otherwise reject and continue.    
        
        # Plot occasionally.
        if i % 100000 == 0:
            plt.scatter(x[:, 0], x[:, 1], c='gray', alpha=0.3,
                        label='data')
            plt.scatter(y_opt[:, 0], y_opt[:, 1], c='limegreen',
                        label='sp(data)')
            plt.scatter(y_t[:, 0], y_t[:, 1], c='red', alpha=1,
                        label='~sp(data)', marker='+')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('iter {}, e={:.4f}'.format(i, energy_t))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig('../output/priv_sp_sample_gridwalk.png')
            #plt.show()
            plt.close()
    print('exited loop')
    pdb.set_trace()
    return y_t, energy_t
    
    
def mixture_model_likelihood(x, y_tilde, bandwidth, do_log=True, tag='',
                             plot=False):
    """Computes likelihood of data set, given cluster centers and bandwidth.

    Args:
      x: NumPy array of raw, full data set.
      y_tilde: NumPy array of cluster/kernel centers.
      bandwidth: Scalar bandwidth of kernels.
      do_log: Boolean, choose to do computations in log scale.
      tag: String, in graph title.
      plot: Boolean, chooses to plot.

    Returns:
      likelihood: Scalar likelihood value for given data set.
    """
    dim = x.shape[1]

    # Set up spherical Gaussian distribution functions centered on
    # each point in the noisy support point set.
    gaussians = [
        multivariate_normal(y, bandwidth * np.eye(dim)) for y in y_tilde]

    def pt_likelihood(pt):
        # (Log-)Likelihood of single point on mixture of Gaussians.
        lik_per_gaussian = [gauss.pdf(pt) for gauss in gaussians]
        lik = 1. / len(gaussians) * sum(lik_per_gaussian)
        return lik, lik_per_gaussian

    liks = []
    lliks = []
    gmm_component_liks = []
    
    for pt in x:
        lik, lik_per_gaussian = pt_likelihood(pt)
        
        try:
            liks.append(lik)
            lliks.append(np.log(lik))
        except:
            pass
        sort_lik = sorted(lik_per_gaussian, reverse=True)
        gmm_component_liks.append(sort_lik)
            
    prod_liks = np.prod(liks)
    sum_lliks = np.sum(lliks)
    likelihood = sum_lliks if do_log else prod_liks
    
    
    print('\t prod_liks={:3.3e},\n\t log_prod_liks={:3.3e},\n\t sum_lliks={:3.3e}\n\n'.format(
        prod_liks, np.log(prod_liks), sum_lliks))
    if prod_liks == 0. and np.log(prod_liks) == -np.Inf:
        pass
    elif not np.isclose(np.log(prod_liks), sum_lliks):
        print('\t [!] Check sum_lliks computation')
        #pdb.set_trace()
        pass

    if likelihood == np.Inf:
        pdb.set_trace()
        
        
    if plot:
        # Plot likelihood of all components for each data point x.
        gmm_component_liks = np.array(gmm_component_liks)
        xs = np.arange(gmm_component_liks.shape[1]).reshape(-1)
        #print(gmm_component_liks)
        for pt_component_lik in gmm_component_liks:
            plt.plot(xs, pt_component_lik, marker=".")    
        low = np.min(gmm_component_liks)
        high = np.max(gmm_component_liks)

        #plt.title('{} gmm component likelihoods: sorted, point-wise\n M={}, bw={:.5f}'.format(
        #    tag, len(x), bandwidth))
        plt.xlabel('gmm components, sorted by lik', fontsize=14)
        plt.ylabel('lik', fontsize=14)
        plt.savefig('../output/mle_gmm_components.png')
        plt.show()

        # Plot histogram of point likelihoods.
        plt.hist(liks)
        plt.title("likelihoods point-wise, bw={:.5f}, prod_lik={:3.3e}".format(
            bandwidth, prod_liks))
        plt.show()


    return likelihood, do_log


def sample_full_set_given_bandwidth(e_opt, energy_sensitivity, x, y_opt,
                                    alpha, bandwidth, sample_size, plot=False,
                                    tag='', method='mh',
                                    power=None, set_seed=False):
    """Samples one full-size data set, given a bandwidth.

    Args:
      e_opt: Energy distance between optimal support points and data.
      energy_sensitivity: Sensitivity, based on user-selected privacy budget.
      x: NumPy array, data.
      y_opt: NumPy array, optimal support points.
      alpha: Float, privacy budget.
      bandwidth: Float, standard deviation of kernel density estimator.
      sample_size: Int, size of expanded sample.
      plot: Boolean, whether to plot.
      tag: String, names plot file.
      method: String, ['diffusion', 'mh'].
      power: Int, power of energy metric. [1, 2]
      set_seed: Boolean, if True, set seed before sample_sp_exp_mech.

    Return:
      y_tilde: NumPy array of sampled, private support points.
      y_tilde_upsampled: NumPy array of upsampled points before adding noise.
      y_tilde_expansion: NumPy array of upsampled points after adding noise.
    """
    ys, es, _ = sample_sp_exp_mech(e_opt, energy_sensitivity, x, y_opt,
                                   method=method, num_y_tildes=1,
                                   alpha=alpha,
                                   power=power,
                                   burnin=5000,
                                   set_seed=set_seed)
    y_tilde = ys[0]

    # Sample from mixture model centered on noisy support points.
    if isinstance(sample_size, float):
        print(len(y_tilde), sample_size)
        pdb.set_trace()
    choices = np.random.choice(range(len(y_tilde)), size=sample_size)
    y_tilde_upsampled = y_tilde[choices]
    y_tilde_upsampled_with_noise = (
        y_tilde_upsampled + np.random.normal(0, bandwidth,
                                             size=(sample_size, x.shape[1])))
    y_tilde_expansion = y_tilde_upsampled_with_noise
    y_tilde_expansion = np.clip(y_tilde_expansion, 0, 1)
    
    # Optionally plot results.
    if plot:
        plt.scatter(x[:, 0], x[:, 1], c='gray', alpha=0.3, label='data')
        plt.scatter(y_tilde[:, 0], y_tilde[:, 1], c='red', alpha=1, 
                    label='~sp(data)', marker='+')
        plt.scatter(y_tilde_expansion[:, 0], y_tilde_expansion[:, 1], c='blue', 
                    alpha=0.3, label='FULL')

        #plt.title('Diffusion, and PRE-SELECTED w = {:.5f}'.format(BANDWIDTH))
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig('../output/fig_kde.png')
        #plt.show()


    return y_tilde, y_tilde_upsampled, y_tilde_expansion


def sp_resample_known_distribution(known_dist_fn, M=100, N=10, DIM=2, 
                                   max_iter=500, learning_rate=1e-2,
                                   power=1., save_iter=[100],
                                   clip=False):
    """Optimizes SP by resampling from known distribution each iter.
    
    Args:
      known_dist_fn: Function that samples data from known distribution.
      M: Scalar, number of data points to sample.
      N: Scalar, number of support points to sample.
      max_iter: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      clip: Boolean, chooses to clip SP to bounded range [0, 1].
    
    Returns:
      sp: NumPy array of support points.
      e_: Scalar energy value associated with those SP.
    """

    # Initialize support points in radial configuration.
    positions = np.linspace(0,
                            (2 * np.pi) * N / (N + 1),
                            N)
    y = [[0.5 + 0.25 * np.cos(rad), 0.5 + 0.25 * np.sin(rad)] for
         rad in positions]
    gen = np.array(y)
    
    # Create (M,1) NumPy array with Exponential random weight for each data point.
    exp_weights = np.random.exponential(scale=1,
                                        size=(M, 1)).astype(np.float32)

    # Set up container for results.
    y_out = np.zeros((max_iter, N, DIM))

    # Build TensorFlow graph.
    tf.reset_default_graph()
    tf_input_data = tf.placeholder(tf.float32, [M, DIM], name='input_data')
    tf_candidate_sp = tf.Variable(gen, name='sp', dtype=tf.float32)

    tf_e_out, _ = energy(tf_input_data, tf_candidate_sp, power=power,
                         is_tf=True, weights=exp_weights)

    opt = tf.train.AdamOptimizer(learning_rate)
    tf_grads, tf_variables = zip(*opt.compute_gradients(tf_e_out))
    tf_optim = opt.apply_gradients(zip(tf_grads, tf_variables))

    # Initialize graph.
    tf_init_op = tf.global_variables_initializer()
    tf_gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf_sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=tf_gpu_options)

    # Run training.
    with tf.Session(config=tf_sess_config) as sess:
        sess.run(tf_init_op)

        start_time = time.time()

        for it in range(max_iter):
            data = known_dist_fn(M)
            data_, sp_, e_, e_grads_, e_vars_ = sess.run(
                [tf_input_data, tf_candidate_sp, tf_e_out, tf_grads,
                 tf_variables],
                {tf_input_data: data})
            sess.run([tf_optim], {tf_input_data: data})

            # Store result in container.
            if clip:
                sp_ = np.clip(sp_, 0, 1)
            y_out[it, :] = sp_
            
            # Plot occasionally.
            #if data.shape[1] == 2 and it in save_iter and it > 0:
            if 0:
                if it > 0:
                    print('  [*] Overall it/s: {:.4f}'.format(
                        (time.time() - start_time) / it))

                plt.scatter(data[:, 0], data[:, 1], c='gray', s=64,
                            alpha=0.3, label='data')
                plt.scatter(sp_[:, 0], sp_[:, 1], s=32, c='limegreen',
                            label='sp')

                # Plot arrows of gradients at each point.
                #for i, sp in enumerate(sp_):
                #    plt.arrow(sp[0], sp[1], *(-1. * e_grads_[0][i]),
                #              color='white', head_width=0.02,
                #              head_length=0.02, length_includes_head=False)

                plt.title('it: {}, e_out: {:.8f}'.format(it, e_))
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                #plt.xlim(0, 1)
                #plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')
                #plt.show()

    time_elapsed = time.time() - time_start
    print('  Time elapsed: {}'.format(time_elapsed))                
                
    sp = y_out[-1]

    return sp, e_


def scatter_and_hist(x, y_all):
    # Isolate data to plot.
    plot_x1 = x[:, 0]
    plot_x2 = x[:, 1]
    plot_y1 = y_all[:, 0]
    plot_y2 = y_all[:, 1]

    # Definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # Start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # Make scatter plot:
    ax_scatter.scatter(plot_x1, plot_x2, color='gray', alpha=0.3,
                       label='data', s=128)
    ax_scatter.scatter(plot_y1, plot_y2, color='limegreen', alpha=0.3,
                       label='sample', s=32)
    #ax_scatter.set_xlim((0, 1))
    #ax_scatter.set_ylim((0, 1))
    ax_scatter.legend()

    # Make histograms.
    ax_histx.hist(plot_x1, bins=20, alpha=0.3, color='gray', label='data',
                  density=True)
    ax_histx.hist(plot_y1, bins=20, alpha=0.3, color='limegreen', label='sample',
                  density=True)
    ax_histy.hist(plot_x2, bins=20, alpha=0.3, orientation='horizontal',
                  color='gray', label='data', density=True)
    ax_histy.hist(plot_y2, bins=20, alpha=0.3, orientation='horizontal',
                  color='limegreen', label='sample', density=True)
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histx.legend()
    ax_histy.legend()

    #plt.show()
    
    
def eval_uncertainty(sampling_fn, M, N, DIM, LR, MAX_ITER,
                     IS_TF, num_draws, plot=False):
    """Measures uncertainty for samples of SP-WLB"""
    
    # Sample fixed data set.
    x = sampling_fn(M)

    # Set up container for results of each draw.
    y_opt_all = np.zeros((num_draws, N, DIM))

    # Take several draws of support points.
    for i in range(num_draws):
        y_opt, e_opt = get_support_points(x, N, MAX_ITER, LR,
                                          is_tf=IS_TF,
                                          Y_INIT_OPTION='random',
                                          clip='data',
                                          do_wlb=True,
                                          plot=plot)
        
        # Store this draw.
        y_opt_all[i] = y_opt

    # Collect support points from all draws.
    y_all = np.concatenate(y_opt_all, axis=0)

    # Plot them.
    scatter_and_hist(x, y_all)
    
    # Show results for FULL SAMPLE.
    print('mean(x) = {}, mean(y) = {}'.format(np.mean(x, axis=0),
                                              np.mean(y_all, axis=0)))
    print('cov(x) =')
    print(np.cov(x, rowvar=False))
    print('cov(y_all) =')
    print(np.cov(y_all, rowvar=False))
    
    
    
    # Show UNCERTAINTY around specific estimations, e.g. mean, variance.
    elements = range(DIM)
    for element in elements:
        xe = x[:, element]
        xe_mean = np.mean(xe)
        ye_means = [np.mean(y, axis=0)[element] for y in y_opt_all]
        plt.hist(ye_means, density=True, color='limegreen', label='estimate',
                 alpha=0.3)
        plt.axvline(x=xe_mean, color='gray', label='data')
        plt.legend()
        plt.title('Marginal mean estimation, element {}'.format(element))
        #plt.show()
