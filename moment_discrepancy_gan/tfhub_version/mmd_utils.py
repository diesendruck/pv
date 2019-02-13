# This script contains helpful distribution metrics and measures.
#
# Note: In many cases, functions can be applied to either TensorFlow or NumPy
#   inputs - a flag toggles this feature - but many of the NumPy versions have
#   not yet been updated to match the TensorFlow versions. To use with NumPy,
#   verify that the NumPy version matches the intended behavior.

import math
import pdb
import numpy as np
import sys
sys.path.append('/home/maurice/mmd')
import tensorflow as tf
#from kl_estimators import naive_estimator as naive_kl
#from kl_estimators import scipy_estimator as scipy_kl
#from kl_estimators import skl_estimator as skl_kl
from scipy.spatial.distance import pdist, cdist


def minmax(x, label=''):
    print('MIN/MAX: {:.2f}, {:.2f}, {}'.format(np.min(x), np.max(x), label))


def exp_taylor(x, k, use_tf=False):
    """Taylor expansion of exp(x), using k terms.
    
    Args:
      x: ndarray of points to apply exponential.
      k: Int, order of Taylor expansion.
      use_tf: Boolean, chooses TF or NumPy
    
    Returns:
      r: ndarray of approximations of exponential applied to each point.
    """
    if use_tf:
        #r = tf.zeros(x.shape)
        #for n in range(k + 1):
        #    r = r + tf.pow(x, n) / math.factorial(n)
        #r = tf.ones(x.shape)
        r = tf.ones(tf.shape(x))
        current = r
        for n in range(1, k + 1):
            current = current * x / n
            r += current
    else:
        #r = np.zeros(x.shape)
        #for n in range(k + 1):
        #    r = r + np.power(x, n) / math.factorial(n)
        r = np.ones(x.shape)
        current = r
        for n in range(1, k + 1):
            current = current * x / n
            r += current
    return r


def get_mmd_from_K(K, n1, n2, use_tf=False):
    num_combos_x = tf.to_float(n1 * (n1 - 1) / 2)
    num_combos_y = tf.to_float(n2 * (n2 - 1) / 2)
    num_combos_xy = tf.to_float(n1 * n2)

    K_xx = K[:n1, :n1]
    K_yy = K[n1:, n1:]
    K_xy = K[:n1, n1:]

    if use_tf:
        K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
        K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_x +
               tf.reduce_sum(K_yy_upper) / num_combos_y -
               2 * tf.reduce_sum(K_xy) / num_combos_xy)
    else:
        K_xx_upper = np.triu(K_xx)
        K_yy_upper = np.triu(K_yy)
        mmd = (np.sum(K_xx_upper) / num_combos_x +
               np.sum(K_yy_upper) / num_combos_y -
               2 * np.sum(K_xy) / num_combos_xy)

    return mmd


def compute_mmd(
        arr1, arr2, sigma_list=None, use_tf=False, slim_output=False,
        kernel_choice='rbf_gaussian'):
    """Computes mmd between two numpy arrays of same size."""
    assert kernel_choice in ['rbf_gaussian', 'rbf_laplace'], \
        'kernel_choice must be in [\'rbf_gaussian\', \'rbf_laplace\']'
    
    if sigma_list is None:
        sigma_list = [1.0]

    if use_tf:
        #n1 = arr1.get_shape().as_list()[0]
        #n2 = arr2.get_shape().as_list()[0]
        n1 = tf.shape(arr1)[0]
        n2 = tf.shape(arr2)[0]

        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        v_sq = tf.reshape(tf.diag_part(VVT), [-1, 1])
        #v_sq_tiled = tf.tile(v_sq, [1, v_sq.get_shape().as_list()[0]])
        v_sq_tiled = tf.tile(v_sq, [1, tf.shape(v_sq)[0]])
        v_sq_tiled_T = tf.transpose(v_sq_tiled)

        exp_obj_gaussian = v_sq_tiled - 2 * VVT + v_sq_tiled_T
        exp_obj_laplace = tf.sqrt(exp_obj_gaussian + 1e-8)  # tf.sqrt gives NaN gradients for zeros.
        #pairwise_difs = tf.expand_dims(v, 1) - tf.expand_dims(v, 0)
        #exp_obj_laplace = tf.norm(pairwise_difs + 1e-8, axis=2)
        if kernel_choice == 'rbf_gaussian':
            exp_object = exp_obj_gaussian
        elif kernel_choice == 'rbf_laplace':
            exp_object = exp_obj_laplace

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
        v_sq = np.reshape(np.diag(VVT), [-1, 1])
        v_sq_tiled = np.tile(v_sq, np.transpose(v_sq).shape)  # Horizontal.
        v_sq_tiled_T = np.transpose(v_sq_tiled)  # Vertical.

        exp_obj_gaussian = v_sq_tiled - 2 * VVT + v_sq_tiled_T
        exp_obj_laplace = np.sqrt(exp_obj_gaussian)
        if kernel_choice == 'rbf_gaussian':
            exp_object = exp_obj_gaussian
        elif kernel_choice == 'rbf_laplace':
            exp_object = exp_obj_laplace
        
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
    polynomial_kernel_a = 1e0  # Constant used in polynomial kernel.
    polynomial_kernel_c = 1e0  # Constant used in polynomial kernel.

    if sigma_list is None:
        sigma_list = [1.0]

    if use_tf:
        #n1 = arr1.get_shape().as_list()[0]
        #n2 = arr2.get_shape().as_list()[0]
        n1 = tf.shape(arr1)[0]
        n2 = tf.shape(arr2)[0]

        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        v_sq = tf.reshape(tf.diag_part(VVT), [-1, 1])
        #v_sq_tiled = tf.tile(v_sq, [1, v_sq.get_shape().as_list()[0]])
        v_sq_tiled = tf.tile(v_sq, [1, tf.shape(v_sq)[0]])
        v_sq_tiled_T = tf.transpose(v_sq_tiled)

        if kernel_choice == 'poly':
            K = tf.pow(polynomial_kernel_a * VVT + polynomial_kernel_c, k_moments)
            
        elif kernel_choice == 'rbf_taylor':
            exp_object = v_sq_tiled - 2 * VVT + v_sq_tiled_T 
            #K = tf.zeros(exp_object.shape)
            K = tf.zeros(tf.shape(exp_object))
            for sigma in sigma_list:
                gamma = 1.0 / (2.0 * sigma**2)
                
                # Since e^x = (e^(x/a))^a, approximate e^x with (taylor(x/a))^a.
                taylor_arg = -gamma * exp_object 
                factor = tf.reduce_max(tf.abs(taylor_arg))
                approx = exp_taylor(taylor_arg / factor, k_moments, use_tf=True) ** factor  # Possibly fix denom
                K += approx
                
                #K += exp_taylor(-gamma * exp_object, k_moments, use_tf=True)
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
            K = np.power(polynomial_kernel_a * VVT + polynomial_kernel_c, k_moments)
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
            K = np.zeros(exp_object.shape)
            for sigma in sigma_list:
                gamma = 1.0 / (2.0 * sigma**2)
                
                # Since e^x = (e^(x/a))^a, approximate e^x with (taylor(x/a))^a.
                taylor_arg = -gamma * exp_object 
                factor = np.max(np.abs(taylor_arg))
                approx = exp_taylor(taylor_arg / factor, k_moments) ** factor
                true = np.exp(-gamma * exp_object)
                K += approx

                #K += exp_taylor(taylor_arg, k_moments)
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
        d_moments = [d_mean]
        for i in range(2, k_moments + 1):
            d_moment_i = np.mean(np.power(d - d_mean, i), axis=0)
            d_moments.append(d_moment_i)
        return d_moments


def compute_moments(arr, k_moments=2):
    """Computes k moments of a NumPy array.
    
    Args:
      arr: Array, shape=(n, d). n points, each d-dimensional.
      k_moments: Number of moments to compute.
    
    Returns:
      m: Array (k, d) of k moments, each d-dimensional.
    """
    #m = []
    m = np.zeros((k_moments, arr.shape[1]))
    for i in range(1, k_moments+1):
        #m.append(np.round(np.mean(arr**i, axis=0), 4))
        m[i - 1] = np.mean(arr**i, axis=0)
    return m


def compute_cmd(arr1, arr2, k_moments=2, use_tf=False, cmd_span_const=None,
        return_terms=False, coordinates=None, taylor_weights=False):
    """Computes Central Moment Discrepancy between two numpy arrays of same size.

    arr1: Data array.
    arr2: Secondary array (possibly simulated).
    k_moments: Integer number of moments to compute.
    use_tf: Boolean, to use TensorFlow instead of NumPy version.
    cmd_span_const: Scalar representing diameter of data.
    return_terms: Boolean, whether to return the list of moment discrepancies.
    coordinates: List, list of coordinates (dimensions) to consider in CMD.
      Set to None, will evaluate all coordinates.
    taylor_weights: Boolean, whether to weight terms in Taylor expansion style.
    """

    # Validate inputs.
    assert arr1.shape[1] == arr2.shape[1], 'arr elems must have same dim'
    assert cmd_span_const is not None, 'define cmd_span_const'

    # Cumulatively sum the CMD, and also collect individual terms.
    cmd = 0
    terms = []

    if use_tf:
        if coordinates is not None:
            arr1 = tf.gather(arr1, coordinates, axis=1)
            arr2 = tf.gather(arr2, coordinates, axis=1)

        arr1_mean = tf.reduce_mean(arr1, axis=0)
        arr2_mean = tf.reduce_mean(arr2, axis=0)

        # First term is just difference of means.
        if taylor_weights:
            weight = 1.
        else:
            weight = cmd_span_const
        first_term = weight * tf.norm(arr1_mean - arr2_mean, ord=2)
        cmd += first_term
        terms.append(first_term)

        for k in range(2, k_moments + 1):
            # Compute k'th central moment, and add to collections.
            # || c_k(arr1 - mean1) - c_k(arr2 - mean2) ||
            if taylor_weights:
                weight = 1. / math.factorial(k)
            else:
                weight = cmd_span_const**k
            term_k = weight * tf.norm(
                tf.reduce_mean(tf.pow(arr1 - arr1_mean, k), axis=0) -
                tf.reduce_mean(tf.pow(arr2 - arr2_mean, k), axis=0))
            cmd += term_k
            terms.append(term_k)

        if return_terms:
            return cmd, terms 
        else:
            return cmd
        
    else:
        # Make a 2D array, [batch_size, data_dim].
        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, [-1, 1])
            arr2 = np.reshape(arr2, [-1, 1])

        # Pick only coordinates of interest.
        if coordinates is not None:
            print('validate coordinate indexing')
            pdb.set_trace()
            arr1 = arr1[:, coordinates]
            arr2 = arr2[:, coordinates]
            pdb.set_trace()

        arr1_mean = np.mean(arr1, axis=0)
        arr2_mean = np.mean(arr2, axis=0)

        first_term = cmd_span_const * np.linalg.norm(arr1_mean - arr2_mean)
        cmd += first_term
        terms.append(first_term)

        for k in range(2, k_moments + 1):
            # Compute k'th moment, and add to collections.
            if taylor_weights:
                weight = 1. / math.factorial(k)
            else:
                weight = cmd_span_const**k
            term_k = weight * np.linalg.norm(
                np.mean(np.power(arr1 - arr1_mean, k), axis=0) -
                np.mean(np.power(arr2 - arr2_mean, k), axis=0))
            cmd += term_k
            terms.append(term_k)

        if return_terms:
            return cmd, terms
        else:
            return cmd


def dp_sensitivity_to_expectation(arr, fn):
    """Computes differential privacy sensitivity for an expectation function.

    arr: Numpy array.
    fn: An expectation function over the entire array, returning a value of dim
      arr.shape[1].

    """

    fn_vals = np.zeros(arr.shape[0])
    for i in range(len(arr)):
        arr_less_i = np.delete(arr, i, axis=0)
        fn_vals[i] = fn(arr_less_i)

    sensitivity = np.max(fn_vals) - np.min(fn_vals)
    return sensitivity


def compute_kth_central_moment(arr, k, use_tf=False):
    assert k >= 2, 'k must be >= 2'
    if use_tf:
        arr_mean = tf.reduce_mean(arr, axis=0)
        result = tf.reduce_mean(tf.pow(arr - arr_mean, k), axis=0)
    else:
        arr_mean = np.mean(arr, axis=0)
        result = np.mean(np.power(arr - arr_mean, k), axis=0)
    return result


def compute_noncentral_moment_discrepancy(
        arr1, arr2, k_moments=2, use_tf=False, cmd_span_const=None,
        return_terms=False, batch_id=None, fbo_noisy_moments=None,
        taylor_weights=False):
    """Computes Noncentral Moment Discrepancy between two arrays of same size.

    arr1: Data array.
    arr2: Secondary array (possibly simulated).
    k_moments: Integer number of moments to compute.
    use_tf: Boolean, to use TensorFlow instead of NumPy version.
    cmd_span_const: Scalar representing diameter of data.
    taylor_weights: Boolean, whether to weight terms in Taylor expansion style.
    """

    # Validate inputs.
    assert arr1.shape[1] == arr2.shape[1], 'arr elems must have same dim'
    assert cmd_span_const, 'define cmd_a, cmd_b'
    if ((batch_id is not None) and (fbo_noisy_moments is not None)):
        fbo = True
        noisy_moments = fbo_noisy_moments[batch_id]
    else:
        fbo = False

    # Cumulatively sum the CMD, and also collect individual terms.
    cmd = 0
    terms = []

    if use_tf:
        """
        if fbo:
            arr1_mean = noisy_moments[0]
        else:
            arr1_mean = tf.reduce_mean(arr1, axis=0)
        arr2_mean = tf.reduce_mean(arr2, axis=0)

        if taylor_weights:
            weight = 1.
        else:
            weight = cmd_span_const
        first_term = weight * tf.norm(arr1_mean - arr2_mean, ord=2)
        cmd += first_term
        terms.append(first_term)
        """

        for k in range(1, k_moments + 1):
            # Compute k'th moments, norm their dif, and add to collections.
            #   || m_k(arr1) - m_k(arr2) ||
            if fbo:
                arr1_mk = noisy_moments[k - 1]
            else:
                arr1_mk = tf.reduce_mean(tf.pow(arr1, k), axis=0)
            arr2_mk = tf.reduce_mean(tf.pow(arr2, k), axis=0)
            
            if taylor_weights:
                weight = 1. / math.factorial(k)
            else:
                weight = cmd_span_const**k
            term_k = weight * tf.norm(arr1_mk - arr2_mk)
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

        """
        if fbo:
            arr1_mean = noisy_moments[0]
        else:
            arr1_mean = np.mean(arr1, axis=0)
        arr2_mean = np.mean(arr2, axis=0)

        if taylor_weights:
            weight = 1.
        else:
            weight = cmd_span_const
        first_term = weight * np.linalg.norm(arr1_mean - arr2_mean)
        cmd += first_term
        terms.append(first_term)
        """

        for k in range(1, k_moments + 1):
            # Compute k'th moment, and add to collections.
            if fbo:
                arr1_mk = noisy_moments[k - 1]
            else:
                arr1_mk = np.mean(np.power(arr1, k), axis=0)
            arr2_mk = np.mean(np.power(arr2, k), axis=0)
            
            if taylor_weights:
                weight = 1. / math.factorial(k)
            else:
                weight = cmd_span_const**k
            term_k = weight * np.linalg.norm(arr1_mk - arr2_mk)
            cmd += term_k
            terms.append(term_k)

        if return_terms:
            return cmd, terms
        else:
            return cmd
        

def compute_joint_moment_discrepancy(
        arr1, arr2, k_moments=2, use_tf=False, cmd_span_const=None,
        return_terms=False, batch_id=None, fbo_noisy_jmoments=None,
        taylor_weights=False):
    """Computes joint moment info, by multiplying component-wise moments.

    Args:
      arr1: Data array.
      arr2: Secondary array (possibly simulated).
      k_moments: Integer number of moments to compute.
      use_tf: Boolean, to use TensorFlow instead of NumPy version.
      cmd_span_const: Scalar representing diameter of data.
      return_terms: Boolean, whether to return terms along with sum.
      batch_id: Int, selects which element in the fixed noisy moments to use.
      fbo_noisy_jmoments: Array, fixed batch onetime noisy joint moments of
        data, [batch_size, k_moments, 1].
      taylor_weights: Boolean, whether to weight terms in Taylor expansion style.
    """

    # Validate inputs.
    assert arr1.shape[1] == arr2.shape[1], 'arr elems must have same dim'
    assert cmd_span_const, 'define cmd_a, cmd_b'
    if ((batch_id is not None) and (fbo_noisy_jmoments is not None)):
        fbo = True
        noisy_jmoments = fbo_noisy_jmoments[batch_id]
    else:
        fbo = False

    # Cumulatively sum the CMD, and also collect individual terms.
    cmd = 0
    terms = []

    if use_tf:
        for k in range(1, k_moments + 1):
            if fbo:
                arr1_jmk = noisy_jmoments[k - 1]
            else:
                arr1_jmk = tf.reduce_mean(
                    tf.pow(tf.reduce_prod(arr1, axis=1), k), axis=0)
            arr2_jmk = tf.reduce_mean(
                tf.pow(tf.reduce_prod(arr2, axis=1), k), axis=0)

            # Add k'th term to collection.
            if taylor_weights:
                weight = 1. / math.factorial(k)
            else:
                weight = cmd_span_const**k
            term_k = weight * tf.norm(arr1_jmk - arr2_jmk)
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

        for k in range(1, k_moments + 1):
            if fbo:
                arr1_jmk = noisy_jmoments[k - 1]
            else:
                arr1_jmk = np.mean(np.power(np.prod(arr1, axis=1), k), axis=0)
            arr2_jmk = np.mean(np.power(np.prod(arr2, axis=1), k), axis=0)

            # Add k'th term to collection.
            if taylor_weights:
                weight = 1. / math.factorial(k)
            else:
                weight = cmd_span_const**k
            term_k = weight * np.linalg.norm(arr1_jmk - arr2_jmk)
            cmd += term_k
            terms.append(term_k)

        if return_terms:
            return cmd, terms
        else:
            return cmd


def compute_energy(x, y, use_tf=False, method='linear'):
    """Distances are euclidean.
    See: https://github.com/syrte/ndtest/blob/master/ndtest.py
    """
    # TODO: Do tf version.
    if use_tf == True:
        sys.exit('TensorFlow version not complete.')

    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z


# The following measures the MMD between a sample and the standard Gaussian.
#   Implementation is for TF. From Guy:
#     https://github.com/guywcole/nn/blob/master/mmd.py
def differences(X, Y, m=None, n=None, p=None):
    #p = int(X.shape[1])
    #m = int(X.shape[0])
    #n = int(Y.shape[0])
    p = tf.shape(X)[1]
    m = tf.shape(X)[0]
    n = tf.shape(Y)[0]
    return tf.tile(tf.reshape(X, [m, 1, p]), [1, n, 1]) - tf.tile(tf.reshape(Y, [1, n, p]), [m, 1, 1])
def MMD_vs_Normal_by_filter(X, filters, sigmas=[1.]):
    """Measures difference from sample X to standard Gaussian.
    For a single group, use filters = [X.shape[0], 1].
    """
    assert X.dtype == filters.dtype, 'filters dtype ({}) must match that of data ({})'.format(filters.dtype, X.dtype)
    distances = tf.reduce_sum(tf.square(differences(X, X)), axis=-1)
    p = int(X.shape[1])
    N, num_sets = [int(val) for val in filters.shape]
    try:
        Ns = tf.reshape(tf.reduce_sum(filters, axis=0, keepdims=False), [num_sets, 1])
    except:
        Ns = tf.reshape(tf.reduce_sum(filters, axis=0, keep_dims=False), [num_sets, 1])
    eps = tf.constant(1e-8, dtype=X.dtype)
    mmds_1vN = [[0. for i in range(num_sets)] for j in range(len(sigmas))]
    try:
        X2 = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    except:
        X2 = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    tf_pi = tf.constant(np.pi, dtype=X.dtype)
    for j in range(len(sigmas)):
        sigma = sigmas[j]
        
        a = tf.constant(-.5 * p * np.log(2. * np.pi * sigma ** 2), distances.dtype)
        b = tf.constant(-.5 / sigma ** 2, distances.dtype)
        xx_embeddings = tf.exp(a + b * distances)
        sum_xx_embeddings = tf.diag_part(tf.matmul(tf.matmul(tf.transpose(filters), xx_embeddings), filters))
        mean_xx_embeddings = (tf.reshape(sum_xx_embeddings, [-1, 1]) - Ns * tf.exp(a))/(Ns * (Ns - 1) + eps)
        
        c = -tf.log(tf.sqrt(2. * tf_pi * (1. + tf.square(sigma))))
        d = (-1. / 2.) / (1. + tf.square(sigma))
        sum_xy_embeddings = tf.matmul(tf.transpose(filters), tf.exp(c + d * X2))
        mean_xy_embeddings = sum_xy_embeddings / (Ns + eps)
        
        mean_yy_embeddings = tf.ones_like(mean_xx_embeddings) / tf.sqrt(2. * tf_pi * (2 + tf.square(sigma)))
        
        # Concatenate SquareX, SquareY, and XY terms.
        mmds_1vN[j] = tf.concat([mean_xx_embeddings, mean_yy_embeddings, -2. * mean_xy_embeddings], 1)
    
    mmds_1vN = tf.convert_to_tensor(mmds_1vN)
    # Mo added: Compute sum in order to return MMD as real number.
    result = tf.reduce_sum(mmds_1vN)
    return result


def privacy_risk(arr, loss_type='mmd'):
    """Measures how difficult it is to privatize a data set.
      
      The more the 'shape' of a distribution depends on a single point, the
      harder it is to privatize. Each point i has an effect on the shape,
      measured by the MMD between the full data set, and the full data set
      with point i removed. If the distributions with vs without i are very
      different (high MMD), then point i is important to the original shape,
      and point i is difficult to privatize, i.e. it is an outlier.

      The privacy risk is a fixed description for a given data set.
    """
    n = len(arr)
    omission_risks = np.zeros((n, 1)) 

    for i in range(n):
        arr_less_i = np.delete(arr, i, axis=0)
        if loss_type == 'mmd':
            omission_risks[i] = compute_mmd(arr, arr_less_i, slim_output=True) 
        elif loss_type == 'kl':
            omission_risks[i] = naive_kl(arr, arr_less_i, k=2) 
            test = naive_kl(np.random.normal(0,1,(100, 1)), np.random.normal(4,1,(100, 1)), k=1)
            print(test)
            pdb.set_trace()
            #omission_risks[i] = scipy_kl(arr, arr_less_i, k=2) 
            #omission_risks[i] = skl_kl(arr, arr_less_i, k=2) 
        #elif loss_type == 'log_ratio_hists':
        #    # QUERY FN
        #    query = lambda x: np.mean(x) 
        #    response_arr = query(arr)
        #    response_arr_less_i = query(arr_less_i)
        #    #hist_bin_edges = np.arange(np.floor(min(arr)), np.ceil(max(arr))+1)
        #    #hist_arr, _ = np.histogram(arr, bins=hist_bin_edges)
        #    #hist_arr_less_i, _ = np.histogram(arr_less_i, bins=hist_bin_edges)
        #    omission_risks[i] = np.abs(response_arr - response_arr_less_i)

    privacy_risk = np.max(omission_risks)
    return privacy_risk, omission_risks
