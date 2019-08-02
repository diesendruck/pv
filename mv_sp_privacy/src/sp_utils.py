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
                       Y_INIT_OPTION='radial', clip='bounds', do_wlb=False,
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
            np.random.normal(0, 0.001, size=[num_support, x.shape[1]])
        
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
                                           #save_iter=[max_iter - 1],
                                           save_iter=200,
                                           clip=clip,
                                           do_wlb=do_wlb,
                                           do_mmd=do_mmd,
                                           mmd_sigma=mmd_sigma,
                                           plot=plot)

    # Get last updated set as support points.
    sp = y_opt[-1]

    return sp, e_opt


def optimize_support_points(data, gen, max_iter=500, learning_rate=1e-2,
                            is_tf=False, energy_power=2., save_iter=[100],
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
        #tf_input_data = tf.placeholder(tf.float32, [data.shape[0], data.shape[1]],
        #                               name='input_data')
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
                    
                elif (plot == True and
                      it % save_iter == 0 and
                      #it in save_iter and
                      it > 0
                     ):
                    graph = pd.plotting.scatter_matrix(pd.DataFrame(sp_), figsize=(10,10))
                    plt.suptitle('SP Optimization. It: {}, e={:.6f}'.format(it, e_))
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
    

    return y_out, e_


def energy(data, gen, power=1., is_tf=False, weights=None):
    """Computes abbreviated energy statistic between two point sets.

    The smaller the value, the closer the sets.
    
    Args:
      data: ND NumPy array of any length, e.g. (1000, 2).
      gen: ND NumPy array of any length, e.g. (10, 2).
      power: Exponent in distance metric. Must be >= 1.
      is_tf: Boolean. Selects for TensorFlow functions.
      weights: (M,1) NumPy array with random weight for each data point.
    
    Returns:
      e: Scalar, the energy between the sets.
      gradients_e: Numpy array of energy gradients for each proposal point.
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


def get_energy_sensitivity(data, N, alpha):
    """Computes energy sensitivity.

    Args:
        data (array): Data set, from which dimension will be computed.
        N (int): Number of support points.
        alpha (float): Privacy budget.

    Returns:
        energy_sensitivity (float): Sensitivity value.
    """
    dim = data.shape[1]
    ENERGY_POWER = 2

    # Define energy sensitivity for Exponential Mechanism.
    energy_sensitivity = 2 * dim ** (1. / ENERGY_POWER) * (2 * N - 1) / N ** 2
    
    return energy_sensitivity
    

def sample_sp_exp_mech(e_opt, energy_sensitivity, x, y_opt, method,
                       step_size=1e-1, num_y_tildes=1, alpha=1., plot=False,
                       diffusion_mean=False, do_mmd=False, mmd_sigma=None,
                       burnin=1000, thinning=200):
    """Samples in space of support points.

    Args:
      e_opt: Energy distance between optimal support points and data.
      energy_sensitivity: Sensitivity, based on user-selected privacy budget.
      x: NumPy array, data.
      y_opt: NumPy array, optimal support points.
      method: String, indicates which sampling method to use.
      step_size: Float, amount to step in diffusion or MH.
      num_y_tildes: Samples of support points to draw.
      alpha: Float, privacy budget.
      plot: Boolean, controls plotting.
      diffusion_mean: Picks mean instead of sampling from Exponential.
      do_mmd: Boolean, chooses MMD instead of energy distance.
      mmd_sigma: Float, bandwidth of MMD kernel.
      burnin: Int, number of burned iterations in Metropolis Hastings.
      thinning: Int, thinning gap in Metropolis Hastings.

    Returns:
      y_tildes: NumPy array, sampled support point sets.
      energies: NumPy array, energies associated with sampled sets.
      energy_estimation_errors: NumPy array, relative error of energy approximation.
    """
    if do_mmd:
        print('[*] Using MMD as distribution distance.')
    
    sensitivity_string = ('\nPr(e) = a / (2U) * exp(- a / (2U) * e) '
                          '~ Exp(2U/a) = Exp(2 * {:.4f} / {:.3f}) = '
                          'Exp({:.4f})\n'.format(energy_sensitivity,
                                                 alpha,
                                                 2 * energy_sensitivity / alpha))
    print(sensitivity_string)

    # Sample support points.
    if method == 'diffusion':

        # ---------- ALGORITHM 1: DIFFUSE FROM OPTIMAL ----------
        # Start with copy of optimal support points, then diffuse until
        # energy(true, copy) is at least e_tilde.

        MAX_COUNT_DIFFUSION = 1e6
        
        y_tildes = np.zeros((num_y_tildes, y_opt.shape[0], y_opt.shape[1]))
        energies = np.zeros(num_y_tildes)
        energy_estimation_errors = np.zeros((num_y_tildes, 2))  # [[energy val, error], ...]

        for i in range(num_y_tildes):
            # TODO: Must be {larger than optimal energy distance, positive}.
            # e_tilde = np.abs(np.random.laplace(
            #     scale=2. * energy_sensitivity / alpha))
            if diffusion_mean:
                print('Using diffusion mean:')
                e_tilde = 2. * energy_sensitivity / alpha
            else:
                e_tilde = np.random.exponential(
                    scale=2. * energy_sensitivity / alpha)
            exp_mean = 2. * energy_sensitivity / alpha
            print('\nExp. mean: {:.5f}, e_tilde: {:.5f}'.format(exp_mean, e_tilde))
            
            # Adjust step size depending on e_tilde.
            #if e_tilde > energy_sensitivity:
            factor = e_tilde / exp_mean
            factor = factor if factor <= 1 else 0.5 * factor
            #if factor > 1:
            #    factor *= 5
            #elif factor > 10:
            #    factor *= 10
            #else:
            #    factor *= 1
            
            step_size_adjusted = step_size * factor
            print('  step_size_factor: {:.5f}, adjusted: {} -> {:.5f}'.format(
                factor, step_size, step_size_adjusted))
            
            y_tilde = y_opt.copy()
            energy_y_y_tilde = 0.
            count = 0

            while energy_y_y_tilde < e_tilde and count <= MAX_COUNT_DIFFUSION:
                
                y_tilde += np.random.normal(0, step_size_adjusted, size=y_tilde.shape)
                y_tilde = np.clip(y_tilde, np.min(x), np.max(x))

                if do_mmd:
                    energy_y_y_tilde, _ = mmd(y_opt, y_tilde, sigma=mmd_sigma)
                else:
                    energy_y_y_tilde, _ = energy(y_opt, y_tilde)

                count += 1

            if count > MAX_COUNT_DIFFUSION:
                plt.scatter(x[:, 0], x[:, 1], c='gray', alpha=0.3,
                            label='data')
                plt.scatter(y_opt[:, 0], y_opt[:, 1], c='limegreen',
                            label='sp(data)')
                plt.scatter(y_tilde[:, 0], y_tilde[:, 1], c='red', alpha=0.7,
                            label='~sp(data)')
                plt.title('{}, it: {}, e(Yt, Y*)={:.4f}\n'.format(
                    method, i, energy_y_y_tilde))
                #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.show()
                print(('ERROR: Did not reach e_tilde level: {:.6f} < {:.6f}'
                       '\nIncrease step size or diffusion max_count.').format(
                    energy_y_y_tilde, e_tilde))
                sys.exit()

                
            error = (energy_y_y_tilde - e_tilde) / e_tilde

            print(('  Diffusion count {:5}, e_opt: {:9.6f}, (e(y, y~) - e~) / e~ '
                   '= error%: ({:5.5f} - {:5.5f}) / {:5.5f} = {:5.5f}').format(
                       count, e_opt, energy_y_y_tilde, e_tilde, e_tilde, error))
            
            
            # Append the accepted energy value to a list for later analysis.
            y_tildes[i] = y_tilde
            energies[i] = energy_y_y_tilde
            energy_estimation_errors[i] = [energy_y_y_tilde, error]
            
            if plot:
                plt.scatter(x[:, 0], x[:, 1], c='gray', alpha=0.3,
                            label='data')
                plt.scatter(y_opt[:, 0], y_opt[:, 1], c='limegreen',
                            label='sp(data)')
                plt.scatter(y_tilde[:, 0], y_tilde[:, 1], c='red', alpha=0.7,
                            label='~sp(data)')
                plt.title('{}, it: {}, e(Yt, Y*)={:.4f}\n'.format(
                    method, i, energy_y_y_tilde))
                #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.show()

                
    elif method == 'mh':

        # ----------------- ALGORITHM 2: METROPOLIS HASTINGS -----------------

        # Let Y* be optimal support points. Initialize Y_t as Y*. Let
        # Y_t' = Y_t + random walk noise. For differential privacy level a
        # and sensitivity U, let acceptance ratio of Y_t' be
        #   \gamma = exp(a / (2U) * [e(Y_t, Y*) - e(Y_t', Y*)]).
        # Then accept Y_t' with probability min(1, \gamma).

        # Choose setup for Metropolis-Hastings.
        chain_length = burnin + thinning * num_y_tildes
        print('Running chain. Length={}, Burn={}, Thin={}, Step={}'.format(
                chain_length, burnin, thinning, step_size))

        # Initialize the support points Y_t.
        # y_t = np.random.uniform(size=y_opt.shape)
        y_t = y_opt  # + np.random.normal(scale=0.1, size=y_opt.shape)
        
        if do_mmd:
            energy_init, _ = mmd(y_opt, y_t, sigma=mmd_sigma)
        else:
            energy_init, _ = energy(y_opt, y_t)

        # Create containers for markov chain results.
        y_mh = np.zeros(shape=(chain_length, y_opt.shape[0], y_opt.shape[1]))
        ratios_unthinned = np.zeros(chain_length)
        energies_unthinned = np.zeros(chain_length)
        accepts = []
        
        # Factor for difference of energies in MH step.
        diff_factor = alpha / (2. * energy_sensitivity)
        print('Difference factor: {:.2f}'.format(diff_factor))


        for i in range(chain_length):
            # Add random walk noise to current set of support points.
            y_update = np.random.normal(scale=step_size, size=y_t.shape)
            #y_update_mask = np.tile(
            #    np.round(np.random.uniform(size=[y_t.shape[0], 1])),
            #    [1, y_t.shape[1]])
            #y_t_candidate = y_t + y_update * y_update_mask
            y_t_candidate = y_t + y_update
            
            y_t_candidate = np.clip(y_t_candidate, 0, 1)
            
            if do_mmd:
                energy_t, _ = mmd(y_opt, y_t, sigma=mmd_sigma)
                energy_t_candidate, _ = mmd(y_opt, y_t_candidate, sigma=mmd_sigma)
            else:
                energy_t, _ = energy(y_opt, y_t)
                energy_t_candidate, _ = energy(y_opt, y_t_candidate)

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
            energy_diff = energy_t - energy_t_candidate
            ratios_unthinned[i] = np.exp(diff_factor * energy_diff)
            
            # print('e_t - e_t\' = {:.5f}, ratio = {:6f}'.format(
            #       energy_t - energy_t_candidate, ratios_unthinned[i]))
            # pdb.set_trace()

            # Accept or reject the candidate.
            if np.random.uniform() < ratios_unthinned[i]:
                accepts.append(i)
                accepted_current = True
                y_t = y_t_candidate
                y_mh[i] = y_t_candidate
                energies_unthinned[i] = energy_t_candidate
            else:
                accepted_current = False
                y_mh[i] = y_t
                energies_unthinned[i] = energy_t

            
            # Plot the points.
            if plot and i % int(chain_length / 10) == 0:
                plt.scatter(x[:, 0], x[:, 1], c='gray', alpha=0.3,
                            label='data')
                plt.scatter(y_opt[:, 0], y_opt[:, 1], c='limegreen',
                            label='sp(data)')
                plt.scatter(y_mh[i][:, 0], y_mh[i][:, 1], c='red', alpha=0.7,
                            label='~sp(data)')
                #plt.title(('{}, it: {}, e(Yt, Y*)={:.4f}, '
                #           'ratio={:.3f}, accepted: {}\n').format(
                #               method, i, energies_unthinned[i],
                #               ratios_unthinned[i], accepted_current))
                #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.tight_layout()
                plt.savefig('../output/fig_mh_sample.png')
                plt.show()
                
                print('Acceptance rate: {}'.format(float(len(accepts)) / (i + 1)))
                print('Energy diff: {:.8f}'.format(energy_diff))


        # Thinned results.
        y_tildes = y_mh[burnin::thinning]
        ratios = ratios_unthinned[burnin::thinning]
        energies = energies_unthinned[burnin::thinning]
        # dups = [y_tildes[i] == y_tildes[i-1] for i in range(1, len(y_tildes))]

        energy_estimation_errors = None
        
        # Plot results of markov chain.
        if plot:
            # Plot acceptance ratios.
            plt.plot(ratios)
            #plt.title('accept_ratios, median={:.5f}'.format(np.median(ratios)))
            plt.xlabel('Sample', fontsize=14)
            plt.ylabel('Acceptance ratio', fontsize=14)
            plt.tight_layout()
            plt.savefig('../output/fig_mh_acceptance_ratios.png')
            plt.show()
            
            # Plot traceplot of energies.
            plt.plot(energies)
            #plt.title('energies, median={:.5f}'.format(np.median(energies)))
            plt.xlabel('Sample', fontsize=14)
            plt.ylabel(r'Energy, $e(y, \tilde{y})$', fontsize=14)
            plt.tight_layout()
            plt.savefig('../output/fig_mh_traceplot_energies.png')
            plt.show()

            # Inspect correlation of energies.
            print('Acceptance ratio: {}'.format(len(accepts) / chain_length))
            print('percent steps that improved energy score: {}'.format(
                sum(ratios_unthinned > 1.) / len(ratios_unthinned)))
            plt.acorr(energies, maxlags=10)
            plt.show()

            # Inspect distribution of energies.
            plt.title('Energies with MH, n={}'.format(len(energies)))
            plt.hist(energies, bins=20, alpha=0.3)
            plt.show()

        # --------------------------------------------------------------

    else:
        print('Method not recognized.')

    return y_tildes, energies, energy_estimation_errors


def sample_sp_mmd_dp_bw(x, energy_power, dim, alpha, n, max_iter=5001, lr=5e-4,
                        plot=False, use_mean=False):
    """Compute private support points with MMD and private bandwidth.
    
    Sensitivity is of the median of pairwise distances, which can be the
    span of the hypercube.
    """
    median_pairwise_dists = np.median(pdist(x, 'minkowski', p=energy_power))
    sensitivity = dim ** (1. / energy_power) / 2.
    
    truth = median_pairwise_dists
    plus_minus_one = 2 * np.random.binomial(1, p=0.5) - 1 
    noise_mean = plus_minus_one * sensitivity / alpha
    noise_natural = np.random.laplace(scale=sensitivity / alpha)
    
    if use_mean:
        private_median = truth + noise_mean
        if private_median <= 0:
            # Assign arbitrary floor.
            private_median = max(private_median, 1e-6)
            print('Private median was negative, assigned 1e-6.')
    else:
        private_median = truth + noise_natural
    assert private_median > 0, 'Private median must be > 0.'
    
    temp_y_opt_mmd, temp_mmd_opt = get_support_points(x, n, max_iter, lr,
                                                      is_tf=True,
                                                      plot=plot,
                                                      do_mmd=True,
                                                      mmd_sigma=private_median)

    print('  [*] Alpha: {:.4f}, True median: {:.4f}, Sampled: {:.4f}'.format(
        alpha, median_pairwise_dists, private_median))

    return temp_y_opt_mmd, temp_mmd_opt



# ----------------
# TODO: DEPRECATE.
def mixture_model_likelihood_mus_weights(x, mus, weights, sigma_data):
    """Computes likelihood of data set, given cluster centers and weights.

    Args:
      x: NumPy array of raw, full data set.
      mus: NumPy array of cluster centers.
      weights: NumPy array of cluster weights.
      sigma_data: Scalar bandwidth used to generate data.

    Returns:
      density: Scalar likelihood value for given data set.
    """
    dim = x.shape[1]
    gaussians = [
        multivariate_normal(mu, sigma_data * np.eye(dim)) for mu in mus]

    def pt_likelihood(pt):
        # Summation of weighted gaussians.
        return sum(
            [weight * gauss.pdf(pt) for weight, gauss in zip(weights,
                                                             gaussians)])

    pts_likelihood = np.prod([pt_likelihood(pt) for pt in x])

    return pts_likelihood
# ----------------


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

        plt.title('{} gmm component likelihoods: sorted, point-wise\n M={}, bw={:.5f}'.format(
            tag, len(x), bandwidth))
        plt.xlabel('gmm components, sorted by lik', fontsize=14)
        plt.ylabel('lik', fontsize=14)
        plt.show()

        # Plot histogram of point likelihoods.
        plt.hist(liks)
        plt.title("likelihoods point-wise, bw={:.5f}, prod_lik={:3.3e}".format(
            bandwidth, prod_liks))
        plt.show()


    return likelihood, do_log


def sample_full_set_given_bandwidth(e_opt, energy_sensitivity, x, y_opt,
                                    step_size, alpha, bandwidth,
                                    sample_size, plot=False, tag='',
                                    method='mh', diffusion_mean=False):
    """Samples one full-size data set, given a bandwidth.

    Args:
      e_opt: Energy distance between optimal support points and data.
      energy_sensitivity: Sensitivity, based on user-selected privacy budget.
      x: NumPy array, data.
      y_opt: NumPy array, optimal support points.
      step_size: Float, amount to step in diffusion or MH.
      alpha: Float, privacy budget.
      bandwidth: Float, standard deviation of kernel density estimator.
      sample_size: Int, size of expanded sample.
      plot: Boolean, whether to plot.
      tag: String, names plot file.
      method: String, ['diffusion', 'mh'].
      diffusion_mean: Picks mean instead of sampling from Exponential.

    Return:
      y_tilde: NumPy array of sampled, private support points.
      y_tilde_upsampled: NumPy array of upsampled points before adding noise.
      y_tilde_expansion: NumPy array of upsampled points after adding noise.
    """
    ys, es, _ = sample_sp_exp_mech(e_opt, energy_sensitivity, x, y_opt,
                                   'diffusion', step_size, 1, alpha=alpha,
                                   diffusion_mean=diffusion_mean)
    y_tilde = ys[0]

    # Sample from mixture model centered on noisy support points.
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
        plt.scatter(y_tilde[:, 0], y_tilde[:, 1], c='red', alpha=0.7, 
                    label='~sp(data)')
        plt.scatter(y_tilde_expansion[:, 0], y_tilde_expansion[:, 1], c='blue', 
                    alpha=0.3, label='FULL')

        #plt.title('Diffusion, and PRE-SELECTED w = {:.5f}'.format(BANDWIDTH))
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig('../output/fig_kde.png')
        plt.show()


    return y_tilde, y_tilde_upsampled, y_tilde_expansion


def sp_resample_known_distribution(known_dist_fn, M=100, N=10, DIM=2, 
                                   max_iter=500, learning_rate=1e-2,
                                   energy_power=2., save_iter=[100],
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
                plt.show()

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

    plt.show()
    
    
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
        plt.show()