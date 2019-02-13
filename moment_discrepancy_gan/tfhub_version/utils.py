from __future__ import print_function

import os
import pdb
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # CREATE LOG DIR OR IDENTIFY EXISTING.
    #if config.load_path:
    #    if config.load_path.startswith(config.log_dir):
    #        config.model_dir = config.load_path
    #    else:
    #        if config.load_path.startswith(config.dataset):
    #            config.model_name = config.load_path
    #        else:
    #            config.model_name = "{}_{}".format(config.dataset, config.load_path)
    #else:
    #    # TODO: Should include all typical settings.
    #    if config.tag.startswith('test'):
    #        config.model_name = "{}".format(config.tag)
    #    else:
    #        config.model_name = "time_{}_{}".format(config.tag, get_time())
    if config.do_pretrain:
        config.log_dir = 'logs/pretrain'
    else:
        if config.tag.startswith('test'):
            config.log_dir = 'logs/{}'.format(config.tag)
        else:
            config.log_dir = 'logs/time_{}_{}'.format(config.tag, get_time())

    # DEFINE DATA LOCATION.
    config.data_path = os.path.join('data', config.dataset)

    # CREATE DIRS IF THEY DON'T EXIST.
    for path in ['logs', 'data', config.log_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.log_dir, "params.json")

    print("[*] MODEL dir: %s" % config.log_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
