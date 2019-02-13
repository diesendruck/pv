from __future__ import print_function

import os
import pdb
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from glob import glob
from itertools import chain
from PIL import Image
from scipy.stats import ks_2samp
from tqdm import trange

from data_loader import get_loader
from models import *
from utils import save_image
from mmd_utils import (compute_mmd, compute_kmmd, compute_cmd,
                       compute_joint_moment_discrepancy,
                       compute_noncentral_moment_discrepancy,
                       compute_moments, compute_central_moments,
                       compute_kth_central_moment, MMD_vs_Normal_by_filter,
                       dp_sensitivity_to_expectation)

np.random.seed(123)


def next(loader):
    return loader.next()[0].data.numpy()


def vert(arr):
    return np.reshape(arr, [-1, 1])


def upper(mat):
    return tf.matrix_band_part(mat, 0, -1) - tf.matrix_band_part(mat, 0, 0)


def sum_normed(mat):
    return mat / tf.reduce_sum(mat)


def to_nhwc(image, data_format, is_tf=False):
    if is_tf:
        if data_format == 'NCHW':
            new_image = nchw_to_nhwc(image)
        else:
            new_image = image
        return new_image
    else:
        if data_format == 'NCHW':
            new_image = image.transpose([0, 2, 3, 1])
        else:
            new_image = image
        return new_image


def nhwc_to_nchw(image, is_tf=False):
    if is_tf:
        if image.get_shape().as_list()[3] in [1, 3]:
            new_image = tf.transpose(image, [0, 3, 1, 2])
        else:
            new_image = image
        return new_image
    else:
        if image.shape[3] in [1, 3]:
            new_image = image.transpose([0, 3, 1, 2])
        else:
            new_image = image
        return new_image


def convert_255_to_n11(image):
    ''' Converts pixel values to range [-1, 1].'''
    image = image/127.5 - 1.
    return image


def convert_n11_to_255(image, is_tf=False):
    if is_tf:
        return tf.clip_by_value((image + 1)*127.5, 0, 255)
    else:
        return np.clip((image + 1)*127.5, 0, 255)


def convert_n11_to_01(image):
    return (image + 1) / 2.


def convert_01_to_n11(image):
    return image * 2. - 1.


def convert_01_to_255(image, is_tf=False):
    if is_tf:
        return tf.clip_by_value(image * 255, 0, 255)
    else:
        return np.clip(image * 255, 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def sort_and_scale(arr):
    """Returns CDF style normalization of array."""
    assert len(arr.shape) == 1, 'Array must be one-dimensional.'
    left_bounded_at_zero = arr - np.min(arr) 
    sorted_arr = np.sort(left_bounded_at_zero)
    sorted_and_scaled_zero_to_one = sorted_arr / (np.max(sorted_arr) + 1e-7)
    return sorted_and_scaled_zero_to_one

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader  # keys: 'images',
                                        #       'iterator_init_op',
                                        #       'dataset_size'

        # Discrepancy values.
        self.k_moments = config.k_moments
        self.cmd_span_const = config.cmd_span_const 
        self.do_taylor_weights = config.do_taylor_weights

        # Loading and pretraining.
        self.load_existing = config.load_existing
        self.do_pretrain = config.do_pretrain

        # Dataset choices.
        self.split = config.split
        self.dataset = config.dataset
        self.mnist_class = config.mnist_class

        # Optimization procedure.
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.use_mmd= config.use_mmd
        self.lambda_mmd_setting = config.lambda_mmd_setting
        self.step = tf.Variable(0, name='step', trainable=False)
        self.g_lr = tf.Variable(config.g_lr, name='g_lr', trainable=False)
        self.g_lr_update = tf.assign(
            self.g_lr, tf.maximum(self.g_lr * 0.9, config.lr_lower_boundary),
            name='g_lr_update')

        # Neural network.
        self.z_dim = config.z_dim
        self.num_conv_filters = config.num_conv_filters
        self.filter_size = config.filter_size
        self.use_bias = config.use_bias
        self.data_format = config.data_format
        _, self.height, self.width, self.channel = \
            get_conv_shape(self.data_loader['images'], self.data_format)
        self.scale_size = self.height 
        self.base_size = config.base_size
        log2_scale_size = int(np.log2(self.scale_size))
        log2_base_size = int(np.log2(self.base_size))
        # Convolutions from 64 down to base_size 4, and 2^(6-2), so 4 conv2d's.
        # where last one is separated out, so 3 repeated convolutions.
        self.repeat_num = log2_scale_size - log2_base_size - 1
        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        # Logging.
        self.log_dir = config.log_dir

        self.use_gpu = config.use_gpu
        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.log_dir)

        sv = tf.train.Supervisor(logdir=self.log_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False


    def build_model(self):
        self.z = tf.random_normal(shape=[self.batch_size, self.z_dim])
        #self.z = tf.truncated_normal(shape=[self.batch_size, self.z_dim])

        # Images from loader are NHWC on [0,1].
        self.x = self.data_loader['images']

        # Set up generator and autoencoder functions. Output g is on [-1, 1].
        g, self.g_var = GeneratorCNN(
            self.z, self.base_size, self.num_conv_filters, self.filter_size,
            self.channel, self.repeat_num, self.data_format, reuse=False,
            use_bias=self.use_bias, verbose=False)
        self.g = convert_n11_to_255(g, is_tf=True)

        # TODO: Resize for TFHub encoder.
        #if g.shape.as_list()[1] != self.scale_size:
        #    g = tf.image.resize_nearest_neighbor(
        #        g, (self.scale_size, self.scale_size))
        g = tf.image.resize_nearest_neighbor(self.g, (224, 224))
        x = tf.image.resize_nearest_neighbor(self.x, (224, 224))
        if self.channel == 1:
            g = tf.image.grayscale_to_rgb(g)
            x = tf.image.grayscale_to_rgb(x)

        # Encode both x and g. Both should start on [0,1], size (224, 224).
        tf_enc_out = tfhub_encoder(tf.concat([x, g], 0))
        self.enc_x, self.enc_g = tf.split(tf_enc_out, 2)

        #######################################################################
        # Set up several losses (e.g. discriminator, discrepancy, autoencoder).

        # Subset encoding to only non-zero columns.
        if self.dataset == 'mnist':
            nonzero_indices = np.load('nonzero_indices_mnist_mobilenetv2035224.npy')
        elif self.dataset == 'birds':
            nonzero_indices = np.load('nonzero_indices_birds_mobilenetv2035224.npy')
        elif self.dataset == 'celeba':
            nonzero_indices = np.load('nonzero_indices_celeba_mobilenetv2035224.npy')
        self.enc_x = tf.gather(self.enc_x, nonzero_indices, axis=1)
        self.enc_g = tf.gather(self.enc_g, nonzero_indices, axis=1)
        print('\n\nUsing {} nonzero of {} total features.\n\n'.format(
            self.enc_x.get_shape().as_list()[1],
            tf_enc_out.get_shape().as_list()[1]))

        # LOSS: Maximum mean discrepancy.
        # Kernel on encodings.
        arr1 = self.enc_x 
        arr2 = self.enc_g 
        sigma_list = [0.0001, 0.001, 0.1]

        data_num = tf.shape(arr1)[0]
        gen_num = tf.shape(arr2)[0]
        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, [1, tf.shape(sqs)[0]])
        exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        self.K = K
        K_xx = K[:data_num, data_num:]
        K_yy = K[data_num:, data_num:]
        K_xy = K[:data_num, data_num:]
        K_xx_upper = upper(K_xx)
        K_yy_upper = upper(K_yy)
        num_combos_xx = tf.to_float(data_num * (data_num - 1) / 2)
        num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)
        num_combos_xy = tf.to_float(data_num * gen_num)

        # Compute and choose between MMD values.
        self.mmd2 = (
            tf.reduce_sum(K_xx_upper) / num_combos_xx +
            tf.reduce_sum(K_yy_upper) / num_combos_yy -
            2 * tf.reduce_sum(K_xy) / num_combos_xy)

        # LOSS: Maximum mean discrepancy, laplace kernel.
        k_moments = self.k_moments
        do_taylor_weights = self.do_taylor_weights
        cmd_span_const = self.cmd_span_const

        self.mmd2_laplace = compute_mmd(
            arr1, arr2, use_tf=True, slim_output=True,
            sigma_list=[0.1, 0.5, 1.0, 2.0],
            kernel_choice='rbf_laplace')


        # LOSS: K-MMD, Taylor expansion.
        self.kmmd = compute_kmmd(
            arr1, arr2, k_moments=k_moments, sigma_list=[0.1, 0.5, 1.0, 2.0],
            use_tf=True, slim_output=True)


        # LOSS: Central moment discrepancy, with k moments.
        self.cmd_k, self.cmd_k_terms = compute_cmd(
            arr1, arr2, k_moments=k_moments, use_tf=True,
            cmd_span_const=cmd_span_const, return_terms=True,
            taylor_weights=do_taylor_weights)


        # LOSS: Noncentral moment discrepancy, with k moments.
        self.ncmd_k = compute_noncentral_moment_discrepancy(
            arr1, arr2, k_moments=k_moments, use_tf=True,
            cmd_span_const=cmd_span_const, taylor_weights=do_taylor_weights)
        _, ncmd_k_terms = compute_noncentral_moment_discrepancy(
            arr1, arr2, k_moments=k_moments, use_tf=True,
            return_terms=True, cmd_span_const=1)  # No coefs, just terms.


        # LOSS: Joint noncentral moment discrepancy, with k moments.
        self.jmd_k = compute_joint_moment_discrepancy(
            arr1, arr2, k_moments=k_moments, use_tf=True,
            cmd_span_const=cmd_span_const, taylor_weights=do_taylor_weights)


        ##############################################################
        # Assemble losses into "final" nodes, to be used by optimizer.

        self.lambda_mmd = tf.Variable(0., trainable=False, name='lambda_mmd')

        if self.dataset == 'mnist':
            #self.g_loss = self.mmd2
            self.g_loss = self.ncmd_k

        elif self.dataset == 'birds':
            #self.g_loss = self.ncmd_k 
            self.g_loss = self.mmd2

        elif self.dataset == 'celeba':
            #self.g_loss = self.ncmd_k 
            self.g_loss = self.mmd2 


        # Optimizer nodes.
        if self.optimizer == 'adam':
            g_opt = tf.train.AdamOptimizer(self.g_lr)
            
        elif self.optimizer == 'rmsprop':
            g_opt = tf.train.RMSPropOptimizer(self.g_lr)

        elif self.optimizer == 'sgd':
            g_opt = tf.train.GradientDescentOptimizer(self.g_lr)


        # Set up optim nodes.
        norm_gradients = 1
        clip = 0
        if norm_gradients:
            # Update_ops node is due to tf.layers.batch_normalization.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                g_grads, g_vars = zip(*g_opt.compute_gradients(
                    self.g_loss, var_list=self.g_var))

                # Replace None with zeros.
                self.g_grads = [g if g is not None else tf.zeros_like(v)
                                for g, v in zip(g_grads, g_vars)]

                ## Normalize each to magnitude 1. 
                #self.g_grads_normed_ = [g / tf.norm(g) for g in self.g_grads]

                # Normalize each by sum of norms.
                g_scalar = tf.maximum(
                    1., tf.reduce_sum([tf.norm(g) for g in self.g_grads]))
                self.g_grads_normed = [g / g_scalar for g in self.g_grads]

                self.g_optim = g_opt.apply_gradients(zip(
                    self.g_grads_normed, g_vars))

        elif clip:
            # Update_ops node is due to tf.layers.batch_normalization.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):

                # CLIP MMD GRADIENTS.
                g_grads, g_vars = zip(*g_opt.compute_gradients(
                    self.g_loss, var_list=self.g_var))
                self.g_grads = g_grads
                g_grads_clipped = tuple(
                    [tf.clip_by_value(g, -0.01, 0.01) for g in g_grads])
                self.g_optim = g_opt.apply_gradients(zip(g_grads_clipped, g_vars))

        else:
            #self.g_optim = g_opt.minimize(
            #    self.g_loss, var_list=self.g_var, global_step=self.step)

            # Update_ops node is due to tf.layers.batch_normalization.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                g_grads, g_vars = zip(*g_opt.compute_gradients(
                    self.g_loss, var_list=self.g_var))
                self.g_grads = [g if g is not None else tf.zeros_like(v)
                                for g, v in zip(g_grads, g_vars)]
                self.g_optim = g_opt.apply_gradients(zip(self.g_grads, g_vars))


        # SUMMARY
        self.summary_op = tf.summary.merge([
            tf.summary.image("a_g", self.g, max_outputs=10),
            tf.summary.image("c_x", self.x, max_outputs=10),
            tf.summary.scalar("loss/mmd2_laplace", self.mmd2_laplace),
            tf.summary.scalar("loss/kmmd", self.kmmd),
            tf.summary.scalar("loss/cmd_k", self.cmd_k),
            tf.summary.scalar("loss/ncmd_k", self.ncmd_k),
            tf.summary.scalar("loss/jmd_k", self.jmd_k),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])


    def generate(self, inputs, root_path=None, step=None, save=False):
        #x = self.sess.run(self.g_read, {self.z_read: inputs})
        #x = self.sess.run(self.g_read, {self.z: inputs})
        x = self.sess.run(self.g, {self.z: inputs})
        if save:
            path = os.path.join(root_path, 'G_{}.png'.format(step))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x


    def encode(self, inputs):
        return self.sess.run(self.encoded_readonly, {self.to_encode_readonly: inputs})


    #def decode(self, inputs):
    #    out = self.sess.run(self.decoded_readonly,
    #        {self.to_decode_readonly: inputs})  # NEEDS to_encode_readonly?
    #    out_unnormed = convert_n11_to_255(out)
    #    return out_unnormed
    def decode(self, dummy_inputs, z_inputs):
        out = self.sess.run(self.decoded_readonly,
            {self.to_encode_readonly: dummy_inputs,
             self.to_decode_readonly: z_inputs})  # NEEDS to_encode_readonly?
        out_unnormed = convert_n11_to_255(out)
        return out_unnormed


    def interpolate_z(self, step, batch_train):
        # Interpolate z when sampled from noise.
        z1 = np.random.normal(0, 1, size=(1, self.z_dim))
        z2 = np.random.normal(0, 1, size=(1, self.z_dim))
        num_interps = 10
        proportions = np.linspace(0, 1, num=num_interps)
        zs = np.zeros(shape=(num_interps, self.z_dim))
        gens = np.zeros([num_interps, self.scale_size, self.scale_size, self.channel])
        for i in range(num_interps):
            zs[i] = proportions[i] * z1 + (1 - proportions[i]) * z2
            gens[i] = self.generate(np.reshape(zs[i], [1, -1]))
        save_image(gens, '{}/interpolate_z_noise{}.png'.format(
            self.log_dir, step))

        # Interpolate between two random encodings.
        #two_random_images = batch_train[:2]
        #im1 = two_random_images[:1, :, :, :]  # This notation keeps dims.
        #im2 = two_random_images[1:, :, :, :]
        #z1 = self.encode(im1)
        #z2 = self.encode(im2)
        #num_interps = 10
        #proportions = np.linspace(0, 1, num=num_interps)
        #zs = np.zeros(shape=(num_interps, self.z_dim))
        #gens = np.zeros([num_interps, self.scale_size, self.scale_size, self.channel])
        #for i in range(num_interps):
        #    zs[i] = proportions[i] * z1 + (1 - proportions[i]) * z2
        #    gens[i] = self.generate(np.reshape(zs[i], [1, -1]))  # Tests generator.
        #save_image(gens, '{}/interpolate_z_enc{}.png'.format(
        #    self.log_dir, step))


    ## Compute Kolmogorov-Smirnov distance between upsampled data weights,
    ##   and gens weights.
    #ks_dist, ks_pval = ks_2samp(g_weights, d_weights_up)
    #ks_dist_, ks_pval_ = ks_2samp(g_weights, d_weights)
    #print('KS(gens, up). dist={:.2f}, p={:.3f}'.format(ks_dist, ks_pval))
    #print('KS(gens, data). dist={:.2f}, p={:.3f}'.format(ks_dist_, ks_pval_))


    def load_checkpoint(self):
        """Restores weights from pre-trained model."""
        import re
        ckpt_path = os.path.join(self.log_dir)
        print(' [*] Reading checkpoints...')
        print('     {}'.format(ckpt_path))
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                               os.path.join(ckpt_path, ckpt_name))
                               #self.log_dir)
            #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            #counter = int(''.join([i for i in ckpt_name if i.isdigit()]))
            counter = int(ckpt_name.split('-')[-1])
            print(' [*] Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            print(' [*] Failed to find a checkpoint')
            return False, 0


    def train(self):
        # Optionally load existing model.
        if self.load_existing:
            could_load, checkpoint_counter = self.load_checkpoint()
            if could_load:
                #load_step = checkpoint_counter
                self.start_step = checkpoint_counter
                print(' [*] Load SUCCESS. Continue? (c)')
                pdb.set_trace()
            else:
                print(' [!] Load FAILED')
                pdb.set_trace()
        else:
            #load_step = 0
            self.start_step = 0
        print('\n{}\n'.format(self.config))
        g_lr_inside = self.config.g_lr

        # Save some fixed images once.
        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        #x_fixed = self.get_images_from_loader()
        #x_fixed = self.sess.run(self.x)
        #save_image(x_fixed, '{}/x_fixed.png'.format(self.log_dir))
        #g_loss_out = self.sess.run(self.g_loss)
        

        # Train generator.
        for step in trange(self.start_step, self.max_step):
            # Reset data loader each epoch.
            num_steps_per_epoch = \
                self.data_loader['dataset_size'] // self.batch_size 
            if step % num_steps_per_epoch == 0:
                self.sess.run(self.data_loader['iterator_init_op'])

            fetch_dict = {
                'g_optim': self.g_optim,
            }

            # Occasionally fetch other nodes for logging/saving.
            if (step % self.log_step == 0):
                fetch_dict.update({
                    'summary': self.summary_op,
                    'g_loss': self.g_loss,
                    'mmd2': self.mmd2,
                    'mmd2_laplace': self.mmd2_laplace,
                    'kmmd': self.kmmd,
                    'cmd_k': self.cmd_k,
                    'ncmd_k': self.ncmd_k,
                    'jmd_k': self.jmd_k,
                    'enc_x': self.enc_x,
                    'enc_g': self.enc_g,
                })

            # For MMDGAN training, use data with predicted weights.
            #batch_train = self.sess.run(self.x)  # NHWC on [0,1]

            #import time
            #t0 = time.time()
            #self.sess.run(self.enc_x)
            #t1 = time.time()
            #print(t1-t0)
            #pdb.set_trace()

            # Run full training step on pre-fetched data and simulations.
            result = self.sess.run(
                fetch_dict,
                feed_dict={
                    #self.x: batch_train,
                    self.lambda_mmd: self.lambda_mmd_setting, 
                    self.g_lr: g_lr_inside,
                    })

            if step % self.lr_update_step == self.lr_update_step - 1:
                g_lr_inside = g_lr_inside * 0.9
                print('\nUpdated g_lr: {}\n'.format(g_lr_inside))

            # Log and save as needed.
            if (step % self.log_step == 0):
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                mmd2 = result['mmd2']
                mmd2_laplace = result['mmd2_laplace']
                kmmd = result['kmmd']
                cmd_k = result['cmd_k']
                ncmd_k = result['ncmd_k']
                jmd_k = result['jmd_k']
                print(('[{}/{}] LOSSES: '
                    ''
                    'mmd2: {:.3f}, mmd2_laplace: {:.3f}, '
                    'kmmd: {:.3f}, cmd_k: {:.3f}, ncmd_k: {:.3f}, jmd_k: {:.3f}, '
                    '').format(
                        step, self.max_step, 
                        mmd2, mmd2_laplace, kmmd, cmd_k, ncmd_k, jmd_k))

                # Save checkpoint.
                ckpt_path = self.log_dir
                self.saver.save(self.sess,
                                os.path.join(ckpt_path, 'model.ckpt'),
                                #self.log_dir,
                                global_step=step)

                # First save a sample.
                #if step == 0:
                #    x_samp = batch_train[:1]  # This indexing keeps dims.
                #    save_image(x_samp, '{}/x_samp.png'.format(self.log_dir))

                # Save images for fixed and random z.
                gen_fixed = self.generate(
                    z_fixed, root_path=self.log_dir, step='fix'+str(step),
                    save=True)
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                gen_rand = self.generate(
                    z, root_path=self.log_dir, step='rand'+str(step),
                    save=True)

                # Save image of interpolation of z.
                #self.interpolate_z(step, batch_train)


                print('ENCODER RANGE: x, g')
                enc_x_ = result['enc_x']
                enc_g_ = result['enc_g']
                print(np.round(np.percentile(enc_x_, [0, 20, 50, 80, 100]), 2))
                print(np.round(np.percentile(enc_g_, [0, 20, 50, 80, 100]), 2))


                # TROUBLESHOOT ENCODING RANGE.
                troubleshoot_encoder = 0
                if troubleshoot_encoder:

                    subset_nonzero = enc_x_[:, np.any(enc_x_ != 0, axis=0)]
                    zero_cols = np.unique(np.where(enc_x_ == 0)[1])
                    nonzero_cols = np.unique(np.where(enc_x_ != 0)[1])
                    def pp(x):
                        return np.percentile(x, [0, 20, 50, 80, 100])
                    pind = np.apply_along_axis(pp, 0, enc_x_)
                    nonzero_pind = np.arange(enc_x_.shape[1])[np.any(pind != 0,
                                                                     axis=0)]
                    set_dif = sorted(list(set(nonzero_cols) - set(nonzero_pind)))
                    subset_nonzero_pp = pind[:, nonzero_pind]

                    print('\n\nTo save nonzero columns of embedding, continue (c).\n\n')
                    pdb.set_trace()
                    if self.dataset == 'mnist':
                        np.save('nonzero_indices_mnist_mobilenetv2035224.npy',
                                nonzero_pind)
                    elif self.dataset == 'birds':
                        np.save('nonzero_indices_birds_mobilenetv2035224.npy',
                                nonzero_pind)
                    elif self.dataset == 'celeba':
                        np.save('nonzero_indices_celeba_mobilenetv2035224.npy',
                                nonzero_pind)

                    # Display percentiles for each nonzero feature.
                    for i in range(subset_nonzero_pp.shape[1]):
                        print(np.round(subset_nonzero_pp[:,i], 2))
                    pdb.set_trace()

                    sys.exit()

                
                # TROUBLESHOOT GENERATOR GRADIENTS.
                troubleshoot_generator= 0
                if troubleshoot_generator:
                    g_loss_out = result['g_loss']
                    g_grads_out1 = self.sess.run(self.g_grads)
                    g_grads_out2 = self.sess.run(self.g_grads_normed_)
                    g_grads_out3 = self.sess.run(self.g_grads_normed)
                    print([np.round(np.max(i), 3) for i in g_grads_out1])
                    print([np.round(np.max(i), 3) for i in g_grads_out2])
                    print([np.round(np.max(i), 3) for i in g_grads_out3])
                    pdb.set_trace()
                    if np.any(np.isnan(g_grads_out[0])):
                        print('mmd grads were NaN')
                        pdb.set_trace()


