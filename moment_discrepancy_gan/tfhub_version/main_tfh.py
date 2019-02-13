"""Autoencoder-based discrepancy GAN, using optionally MMD, CMD, and others."""

import numpy as np
import os
import pdb
import sys
import tensorflow as tf

from trainer_tfh import Trainer
from config import get_config
#from data_loader import get_loader
from dataset_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

def main(config):
    # NOTE: Run this in shell first.
    if np.float(tf.__version__[:3]) < 1.7:
        sys.exit('***NOTE!***: FIRST RUN:\n"source ~/virtualenvironment/tf1.7/bin/activate"')

    prepare_dirs_and_logger(config)

    # Alert if config.log_dir already contains files.
    if not config.load_existing:
        if len(os.listdir(config.log_dir)) > 1:
            print(('log dir ({}) contains files besides checkpoints dir, '
                   'continue? (c)').format(config.log_dir))
            pdb.set_trace()


    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        setattr(config, 'batch_size', 64)
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    if config.dataset == 'mnist':
        #directory_to_load = '7s_train'
        #directory_to_load = '8s_train'
        directory_to_load = '1to9_train'
    elif config.dataset == 'birds':
        directory_to_load = 'images_preprocessed'
    elif config.dataset == 'celeba':
        directory_to_load = 'train'

    inputs = get_loader(
        data_path, config.batch_size, config.scale_size,
        config.data_format, split_name=directory_to_load,
        )
    trainer = Trainer(config, inputs)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
