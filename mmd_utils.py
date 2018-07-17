import math
import pdb
import numpy as np
import tensorflow as tf



def minmax(x, label=''):
    print('MIN/MAX: {:.2f}, {:.2f}, {}'.format(np.min(x), np.max(x), label))


def taylor(x, k, use_tf=False):
    if use_tf:
        r = tf.zeros(x.shape)
        for i in range(k + 1):
            r += tf.pow(x, i) / math.factorial(i)
    else:
        r = np.zeros(x.shape)
        for i in range(k + 1):
            r += np.power(x, i) / math.factorial(i)
    return r


def get_mmd_from_K(K, n1, n2, use_tf=False):
    num_combos_x = n1 * (n1 - 1) / 2
    num_combos_y = n2 * (n2 - 1) / 2

    K_xx = K[:n1, :n1]
    K_yy = K[n1:, n1:]
    K_xy = K[:n1, n1:]

    if use_tf:
        K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
        K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_x +
               tf.reduce_sum(K_yy_upper) / num_combos_y -
               2 * tf.reduce_sum(K_xy) / (n1 * n2))
    else:
        K_xx_upper = np.triu(K_xx)
        K_yy_upper = np.triu(K_yy)
        mmd = (np.sum(K_xx_upper) / num_combos_x +
               np.sum(K_yy_upper) / num_combos_y -
               2 * np.sum(K_xy) / (n1 * n2))

    return mmd


def compute_mmd(arr1, arr2, sigma_list=None, use_tf=False, slim_output=False):
    """Computes mmd between two numpy arrays of same size."""
    if sigma_list is None:
        sigma_list = [1.0]

    if use_tf:
        n1 = arr1.get_shape().as_list()[0]
        n2 = arr2.get_shape().as_list()[0]

        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        v_sq = tf.reshape(tf.diag_part(VVT), [-1, 1])
        v_sq_tiled = tf.tile(v_sq, [1, v_sq.get_shape().as_list()[0]])
        v_sq_tiled_T = tf.transpose(v_sq_tiled)
        exp_object = v_sq_tiled - 2 * VVT + v_sq_tiled_T

        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]
        K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
        K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
        num_combos_x = tf.to_float(n1 * (n1 - 1) / 2)
        num_combos_y = tf.to_float(n2 * (n2 - 1) / 2)
        num_combos_xy = tf.to_float(n1 * n2)
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_x +
               tf.reduce_sum(K_yy_upper) / num_combos_y -
               2 * tf.reduce_sum(K_xy) / num_combos_xy)
        if slim_output:
            return mmd
        else:
            return mmd, exp_object

    else:
        n1 = len(arr1)
        n2 = len(arr2)

        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, [-1, 1])
            arr2 = np.reshape(arr2, [-1, 1])
        v = np.concatenate((arr1, arr2), 0)
        VVT = np.matmul(v, np.transpose(v))
        sqs = np.reshape(np.diag(VVT), [-1, 1])
        sqs_tiled_horiz = np.tile(sqs, np.transpose(sqs).shape)
        exp_object = sqs_tiled_horiz - 2 * VVT + np.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += np.exp(-gamma * exp_object)
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]
        K_xx_upper = np.triu(K_xx)
        K_yy_upper = np.triu(K_yy)
        num_combos_x = n1 * (n1 - 1) / 2
        num_combos_y = n2 * (n2 - 1) / 2
        mmd = (np.sum(K_xx_upper) / num_combos_x +
               np.sum(K_yy_upper) / num_combos_y -
               2 * np.sum(K_xy) / (n1 * n2))
        if slim_output:
            return mmd
        else:
            return mmd, exp_object


def compute_kmmd(arr1, arr2, k_moments=2, kernel_choice='rbf_taylor',
        sigma_list=None, use_tf=False, slim_output=False, verbose=False):
    """Computes k-mmd^2 between two numpy arrays of same size.

    NOTE: Currently, all calculations are for k=2, i.e. a metric that matches
      the first two moments.
    
    """
    poly_c = 1e0  # Constant used in polynomial kernel.
    poly_a = 1e0  # Constant used in polynomial kernel.

    if sigma_list is None:
        sigma_list = [1.0]

    if use_tf:
        n1 = arr1.get_shape().as_list()[0]
        n2 = arr2.get_shape().as_list()[0]

        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        v_sq = tf.reshape(tf.diag_part(VVT), [-1, 1])
        v_sq_tiled = tf.tile(v_sq, [1, v_sq.get_shape().as_list()[0]])
        v_sq_tiled_T = tf.transpose(v_sq_tiled)

        if kernel_choice == 'poly':
            K = tf.pow(poly_a * VVT + poly_c, k_moments)
        elif kernel_choice == 'rbf_taylor':
            exp_object = v_sq_tiled - 2 * VVT + v_sq_tiled_T 
            K = 0.0
            for sigma in sigma_list:
                gamma = 1.0 / (2.0 * sigma**2)
                K += taylor(-gamma * exp_object, k_moments, use_tf=True)
                # For troubleshooting, regular RBF:
                #K += tf.exp(-gamma * exp_object)

        kmmd = get_mmd_from_K(K, n1, n2, use_tf=True)

        if slim_output:
            return kmmd
        else:
            return kmmd, K 
        
    elif not use_tf:
        n1 = len(arr1)
        n2 = len(arr2)

        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, [-1, 1])
            arr2 = np.reshape(arr2, [-1, 1])

        # Stack inputs, get Gram matrix, extract squared terms, tile squares,
        # transpose it, do another outer product.
        v = np.concatenate((arr1, arr2), 0)
        VVT = np.matmul(v, np.transpose(v))
        v_sq = np.reshape(np.diag(VVT), [-1, 1])
        v_sq_tiled = np.tile(v_sq, [1, v_sq.shape[0]])
        v_sq_tiled_T = np.transpose(v_sq_tiled)
        VVT_sq = np.matmul(v_sq, np.transpose(v_sq))

        if kernel_choice == 'poly':
            K = np.power(poly_a * VVT + poly_c, k_moments)
            if verbose:
                K0 = np.power(VVT + c, num_moments)
                K1 = np.exp(np.power(VVT + c, num_moments))
                K2 = -1. * np.power(VVT + c, num_moments)
                K3 = np.exp(-1. * np.power(VVT + c, num_moments))
                K4 = np.power(1e-1 * VVT + c, num_moments)
                print(np.min(K0), np.max(K0))
                print(np.min(K1), np.max(K1))
                print(np.min(K2), np.max(K2))
                print(np.min(K3), np.max(K3))
                print(np.min(K4), np.max(K4))
                pdb.set_trace()

        elif kernel_choice == 'rbf_taylor':
            exp_object = v_sq_tiled - 2 * VVT + v_sq_tiled_T 
            K = 0.0
            for sigma in sigma_list:
                gamma = 1.0 / (2.0 * sigma**2)
                K += taylor(-gamma * exp_object, k_moments)
                # For troubleshooting, regular RBF:
                #K += np.exp(-gamma * exp_object)

        else:
            raise ValueError('kernel choice not in ["poly", "rbf_taylor"')


        kmmd = get_mmd_from_K(K, n1, n2)

        if verbose:
            reg = np.exp(-gamma * exp_object)
            tay = taylor(-gamma * exp_object, 2)
            minmax(exp_object, 'exp_object')
            minmax(-gamma * exp_object, 'exp_object_with_gamma')
            minmax(reg, 'reg')
            minmax(tay, 'tay')
            kmmd_tay = get_mmd_from_K(K, n1, n2)
            print('kmmd: {:.4f}\nkmmd_tay: {:.4f}'.format(kmmd, kmmd_tay))
            pdb.set_trace()

        if slim_output:
            return kmmd
        else:
            return kmmd, K 


def compute_central_moments(d, k_moments=2, use_tf=False):
    """Computes central moments of data in NumPy array."""
    if use_tf:
        d_mean = tf.reduce_mean(d, axis=0)
        d_moments = [d_mean]
        for i in range(2, k_moments + 1):
            d_moment_i = tf.reduce_mean(tf.pow(d - d_mean, i), axis=0)
            d_moments.append(d_moment_i)
        return d_moments

    else:
        d_mean = np.mean(d, axis=0)
        #d_moments = [list(np.round(d_mean, 4))]
        d_moments = [d_mean]
        for i in range(2, k_moments + 1):
            d_moment_i = np.mean(np.power(d - d_mean, i), axis=0)
            d_moments.append(d_moment_i)
        return d_moments


def compute_cmd(arr1, arr2, k_moments=2, use_tf=False, cmd_a=None, cmd_b=None,
        return_terms=False):
    """Computes Central Moment Discrepancy between two numpy arrays of same size.

    cmd_a: Min on data.
    cmd_b: Max on data.
    cmd_gamma: Float. Scales exponent of each CMD term's coefficient. Decreasing
      this from 1 toward 0, slows the decay of weights on the higher-order
      terms, i.e. it becomes more sensitive to higher-order moments.
    
    """
    assert (cmd_a is not None) and (cmd_b is not None), 'define cmd_a, cmd_b'
    span_const = 1. / np.abs(cmd_b - cmd_a)

    # Cumulatively sum the CMD, and also collect individual terms.
    cmd = 0
    terms = []

    if use_tf:
        arr1_mean = tf.reduce_mean(arr1, axis=0)
        arr2_mean = tf.reduce_mean(arr2, axis=0)

        first_term = span_const * tf.norm(arr1_mean - arr2_mean, ord=2)
        cmd += first_term
        terms.append(first_term)

        for k in range(2, k_moments + 1):
            # Compute k'th moment, and add to collections.
            term_k = (span_const**k) * tf.norm(
                tf.reduce_mean(tf.pow(arr1 - arr1_mean, k), axis=0) -
                tf.reduce_mean(tf.pow(arr2 - arr2_mean, k), axis=0))
            cmd += term_k
            terms.append(term_k)

        if return_terms:
            return cmd, terms 
        else:
            return cmd
        
    else:
        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, [-1, 1])
            arr2 = np.reshape(arr2, [-1, 1])

        arr1_mean = np.mean(arr1, axis=0)
        arr2_mean = np.mean(arr2, axis=0)

        first_term = span_const * np.linalg.norm(arr1_mean - arr2_mean)
        cmd += first_term
        terms.append(first_term)

        for k in range(2, k_moments + 1):
            # Compute k'th moment, and add to collections.
            term_k = (span_const**k) * np.linalg.norm(
                np.mean(np.power(arr1 - arr1_mean, k), axis=0) -
                np.mean(np.power(arr2 - arr2_mean, k), axis=0))
            cmd += term_k
            terms.append(term_k)

        if return_terms:
            return cmd, terms
        else:
            return cmd
