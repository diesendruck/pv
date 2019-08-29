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
                       Y_INIT_OPTION='uniform', clip='bounds', do_wlb=False,
                       plot=True, do_mmd=False, mmd_sigma=None):
    """Initializes and gets support points.
    
    Args:
      x: Data. ND numpy array of any length, e.g. (100, dim).
      num_support: Scalar.
      max_iter: Scalar, number of times to loop through updates for all vars.
      lr: Scalar, amount to move point with each gradient update.
      is_tf: Boolean, chooses TensorFlow optimization.
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
        
    print('\nSTARTING RUN. is_tf: {}, y_init: {}, clip: {}, wlb: {}'.format(
        is_tf, Y_INIT_OPTION, clip, do_wlb))
    
    # Initialize generated particles for both sets (y and y_).
    d = x.shape[1]
    offset = 0.1 * (np.max(x) - np.min(x))

    if Y_INIT_OPTION == 'grid':
        grid_size = int(np.sqrt(num_support))
        assert grid_size == np.sqrt(num_support), \
            'num_support must square for grid'
        #_grid = np.linspace(offset, 1 - offset, grid_size)
        _grid = np.linspace(np.min(x) + offset, np.max(x) - offset, grid_size)
        y = np.array(np.meshgrid(_grid, _grid)).T.reshape(-1, d)
        # Perturb grid in order to give more diverse gradients.
        y += np.random.normal(0, 0.005, size=y.shape)

    elif Y_INIT_OPTION == 'random':
        #y = np.random.uniform(offset, 1 - offset, size=(num_support, d))
        offset = 0.1 * (np.max(x) - np.min(x))
        y = np.random.uniform(np.min(x) + offset, np.max(x) - offset,
                              size=(num_support, d))

    elif Y_INIT_OPTION == 'radial':
        positions = np.linspace(0,
                                (2 * np.pi) * num_support / (num_support + 1),
                                num_support)
        y = [[0.5 + 0.25 * np.cos(rad), 0.5 + 0.25 * np.sin(rad)] for
             rad in positions]
        y = np.array(y)
        
    elif Y_INIT_OPTION == 'subset':
        y = x[np.random.randint(len(x), size=num_support)] + \
            np.random.normal(0, 0.0001, size=[num_support, x.shape[1]])
        #y = np.array(x[np.random.randint(len(x), size=num_support)])
        
    elif Y_INIT_OPTION == 'uniform':
        y = np.random.uniform(np.min(x), np.max(x), size=[num_support, x.shape[1]])
        
    # For MMD, must supply sigma.
    if do_mmd:
        assert mmd_sigma is not None, 'If using MMD, must supply mmd_sigma.'

    # Optimize particles for each dataset (x0 and x1).
    y_opt, e_opt = optimize_support_points(x, y, max_iter=max_iter,
                                           learning_rate=lr, is_tf=is_tf,
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
                            is_tf=False, energy_power=2., save_iter=100,
                            clip='bounds', do_wlb=False, do_mmd=False,
                            mmd_sigma=None, plot=True):
    """Runs TensorFlow optimization, n times through proposal points.
    
    Args:
      data: ND numpy array of any length, e.g. (100, dim).
      gen: ND numpy array of any length, e.g. (10, dim).
      max_iter: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      is_tf: Boolean, chooses TensorFlow optimization.
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
        tf_input_data = tf.placeholder(tf.float32, [None, data.shape[1]],
                                       name='input_data')
        tf_candidate_sp = tf.Variable(gen, name='sp', dtype=tf.float32)

        if do_mmd:
            tf_e_out, _ = mmd(tf_input_data, tf_candidate_sp, sigma=mmd_sigma,
                              is_tf=True, weights=exp_weights)
        else:
            tf_e_out, _ = energy(tf_input_data, tf_candidate_sp, power=energy_power,
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
                    plt.close()
                    
                elif (plot == True and
                      it % save_iter == 0 and
                      #it in save_iter and
                      it > 0
                     ):
                    graph = pd.plotting.scatter_matrix(pd.DataFrame(sp_), figsize=(10,10))
                    plt.suptitle('SP Optimization. num_supp={}, it={}, e={:.6f}'.format(len(sp_), it, e_))
                    plt.show()
                    plt.close()
                    
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
                e_, e_grads = energy(data, gen, power=energy_power)
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
                plt.close()
    

    return y_out, e_


def energy(data, gen, power=2., is_tf=False, weights=None, return_k=False):
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


def get_energy_sensitivity(data, num_supp, power=None):
    """Computes energy sensitivity.

    Args:
        data (array): Data set, from which dimension will be computed.
        num_supp (int): Number of support points.

    Returns:
        energy_sensitivity (float): Sensitivity value.
    """
    dim = data.shape[1]

    # Define energy sensitivity for Exponential Mechanism.
    energy_sensitivity = 2 * dim ** (1. / power) * (2 * num_supp - 1) / num_supp ** 2
    
    return energy_sensitivity


def plot_nd(d, w=10, h=10, title=None):
    graph = pd.plotting.scatter_matrix(pd.DataFrame(d), figsize=(w, h));
    if title:
        plt.suptitle(title)
    plt.savefig('../results/regression_logs/plot_nd_{}.png'.format(title))
    plt.show()
    plt.close()


def sample_sp_exp_mech(
        x, num_supp, alpha=None, save_dir=None, plot=False, burnin=5000,
        thinning=2000, power=1, max_iter=301, lr=1e-2):
    """Samples in space of support points.

    Args:
      x: NumPy array, data.
      num_supp: Number of support points per set.
      alpha: Float, privacy budget.
      save_dir: String, location to save plots.
      plot: Boolean, controls plotting.
      burnin: Int, number of burned iterations in Metropolis Hastings.
      thinning: Int, thinning gap in Metropolis Hastings.
      power: Int, power in energy metric.
      max_iter: Int, num iters for sp optim.
      lr: Float, learning rate for sp optim.

    Returns:
      y_tilde: NumPy array, sampled support point sets.
      energy: Float, energy associated with sampled sets.
    """
    num_y_tildes = 1

    # Get optimal support points.
    y_opt, e_opt = get_support_points(x, num_supp, max_iter, lr, is_tf=True)

    # Define energy sensitivity for Exponential Mechanism.
    energy_sensitivity = get_energy_sensitivity(x, num_supp, power=power)
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
    #y_t = np.random.uniform(size=y_opt.shape)
    y_t = y_opt

    # Compute initial value.
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
    
    
    # Run chain.
    for i in range(chain_length):
        
        # Perturb one point at a time.
        update_i = i % len(y_t)
        y_t_candidate = np.copy(y_t)
        y_t_candidate[update_i] += np.random.normal(scale=step_size, size=y_t.shape[1])

        # Clip candidate values to data domain of [0,1].
        y_t_candidate = np.clip(y_t_candidate, 0, 1)


        # Compute energy difference due to single-point change in candidate.
        K_cc_old = [np.linalg.norm(y_t[update_i] - y_t[i], ord=power) for i in range(len(y_t))]
        K_co_old = [np.linalg.norm(y_t[update_i] - y_opt[i], ord=power) for i in range(len(y_opt))]
        
        K_cc_new = [np.linalg.norm(y_t_candidate[update_i] - y_t_candidate[i], ord=power) for i in range(len(y_t))]
        K_co_new = [np.linalg.norm(y_t_candidate[update_i] - y_opt[i], ord=power) for i in range(len(y_opt))]
        
        part_e_old = (2. / (len(y_opt) ** 2)) * (np.sum(K_co_old) - np.sum(K_cc_old))
        part_e_new = (2. / (len(y_opt) ** 2)) * (np.sum(K_co_new) - np.sum(K_cc_new))
        
        
        # Compute the acceptance ratio.
        # With U = 2 * DIM ** (1. / ENERGY_POWER) * (2 * N - 1) / N ** 2
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
        energy_diff = part_e_new - part_e_old
        energy_t_candidate = energy_t + energy_diff
        ratios_unthinned[i] = np.exp(diff_factor * -1. * energy_diff)

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
            print('acceptance_rate={:.8f}, step_size={:.8f}'.format(acceptance_rate, step_size))
            print('Energy diff: {:.8f}'.format(energy_diff))

        # Plot the points.
        if plot and x.shape[1] == 2 and i % int(chain_length / 10) == 0:
            plt.scatter(x[:, 0], x[:, 1], c='gray', alpha=0.3,
                        label='data')
            plt.scatter(y_opt[:, 0], y_opt[:, 1], c='limegreen',
                        label='sp(data)')
            plt.scatter(y_mh[i][:, 0], y_mh[i][:, 1], c='red', alpha=0.7,
                        label='~sp(data)')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'mh_sample.png'))
            plt.show()
            plt.close()
        elif plot and x.shape[1] > 2 and i % int(chain_length / 10) == 0:
            plot_nd(y_mh[i],
                    title='MH ~SP (priv={}, num_supp={}). It: {}, e={:.6f}'.format(
                        alpha, num_supp, i, energies_unthinned[i]))


    # Thinned results.
    y_tildes = y_mh[burnin::thinning]
    ratios = ratios_unthinned[burnin::thinning]
    energies = energies_unthinned[burnin::thinning]

    # Plot results of markov chain.
    if plot:
        # Plot acceptance ratios.
        plt.plot(acceptance_rates)
        #plt.title('accept_ratios, median={:.5f}'.format(np.median(ratios)))
        plt.xlabel('Sample', fontsize=14)
        plt.ylabel('Acceptance rates', fontsize=14)
        plt.ylim((0,1))
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(save_dir, 'mh_acceptance_ratios.png'))
        plt.close()

        # Plot traceplot of energies.
        plt.plot(energies_unthinned)
        #plt.title('energies, median={:.5f}'.format(np.median(energies)))
        plt.xlabel('Sample', fontsize=14)
        plt.ylabel(r'Energy, $e(y, \tilde{y})$', fontsize=14)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(save_dir, 'mh_traceplot_energies.png'))
        plt.close()

        # Inspect correlation of energies.
        print('Acceptance rate: {:.3f}'.format(len(accepts) / chain_length))
        print('percent steps that improved energy score: {:.3f}'.format(
            sum(ratios_unthinned > 1.) / len(ratios_unthinned)))

        # Inspect distribution of energies.
        plt.hist(energies, bins=20, alpha=0.3)
        plt.title('Energies with MH, n={}'.format(len(energies)))
        plt.show()
        plt.close()

    # --------------------------------------------------------------

    return y_tildes[0], energies[0]
