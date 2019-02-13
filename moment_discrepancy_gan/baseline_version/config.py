#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--k_moments', type=int, default=3)
net_arg.add_argument('--scale_size', type=int, default=32, choices=[32, 64],
                     help=('input image will be resized with the given value '
                           'as width and height'))
net_arg.add_argument('--num_conv_filters', type=int, default=16,
                     choices=[2, 4, 6, 8, 16, 32, 64, 128],
                     help='n in the paper')
net_arg.add_argument('--z_dim', type=int, default=16,
                     choices=[1, 2, 3, 4, 5, 6, 10, 16, 32, 64, 128],
                     help='Dimension of hidden layer in autoencoder.')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='mnist',
                      choices=['CelebA', 'mnist'])
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=64)
data_arg.add_argument('--grayscale', type=str2bool, default=True)
data_arg.add_argument('--num_worker', type=int, default=4)
data_arg.add_argument('--target_num', type=int, default=2000,
                      help=('# of target samples, used to sample target group '
                            'mean and covariance'))
data_arg.add_argument('--mnist_class', type=int, default=7,
                      help=('class of MNIST to fetch for self-normalized examples'))

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='rmsprop')
train_arg.add_argument('--max_step', type=int, default=200000)
train_arg.add_argument('--lr_update_step', type=int, default=20000)
train_arg.add_argument('--d_lr', type=float, default=0.0001)
train_arg.add_argument('--g_lr', type=float, default=0.0001)
train_arg.add_argument('--c_lr', type=float, default=0.0001)
train_arg.add_argument('--w_lr', type=float, default=0.0001)
train_arg.add_argument('--lr_lower_boundary', type=float, default=5e-5)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--use_mmd', type=str2bool, default=True)
train_arg.add_argument('--lambda_mmd_setting', type=float, default=0.1)
train_arg.add_argument('--lambda_ae_setting', type=float, default=100.0)
train_arg.add_argument('--weighted', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--tag', type=str, default='test')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--save_step', type=int, default=1000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO',
                      choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help=('directory with images which will be used in test '
                            'sample generation'))
misc_arg.add_argument('--sample_per_image', type=int, default=64,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)


def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed
