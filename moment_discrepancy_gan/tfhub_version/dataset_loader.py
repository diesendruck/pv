import numpy as np
import os
import pdb
import sys
import tensorflow as tf
from glob import glob
from PIL import Image, ImageOps, ImageFilter


def parse_function(filename, scale_size):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    resized_image = tf.image.resize_images(image, [scale_size, scale_size])
    return resized_image


def train_preprocess(image):
    #image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def get_loader(root, batch_size, scale_size, data_format, split_name=None,
               seed=None):
    dataset_name = os.path.basename(root)

    # Chooses dataset to use. 
    root = os.path.join(root, split_name)  # for example, data/birds/train

    # Fetch list of filenames.
    for ext in ["jpg", "png"]:
        filenames = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png
        
        if len(filenames) != 0:
            break
    assert len(filenames) > 0, 'did not find filenames'

    filenames = np.random.permutation(filenames)

    with tf.device('/cpu:0'):
        dataset = (
            tf.data.Dataset.from_tensor_slices(filenames)
                .apply(tf.contrib.data.map_and_batch(
                    lambda x: parse_function(x, scale_size),
                    batch_size=batch_size, num_parallel_calls=8))
                #.map(train_preprocess, num_parallel_calls=10)
                #.batch(batch_size)
                .prefetch(1))
            #tf.data.Dataset.from_tensor_slices(filenames)
            #    .shuffle(buffer_size=len(filenames))
            #    .map(lambda x: parse_function(x, scale_size),
            #         num_parallel_calls=8)
            #    #.map(train_preprocess, num_parallel_calls=10)
            #    .batch(batch_size)
            #    .prefetch(1))

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images,
              'iterator_init_op': iterator_init_op,
              'dataset_size': len(filenames)}

    return inputs


    #filename_queue = tf.train.string_input_producer(
    #    list(filenames), shuffle=False, seed=seed)
    #reader = tf.WholeFileReader()
    #filename, data = reader.read(filename_queue)
    #image = tf_decode(data, channels=channels)
    #if dataset_name != 'mnist':
    #    image = tf.image.random_flip_left_right(image)  # Data augmentation.

    #with Image.open(filenames[0]) as img:
    #    w, h = img.size
    #    shape = [h, w, channels]

    #if grayscale:
    #    image = tf.image.rgb_to_grayscale(image)
    #    image.set_shape([h, w, 1])
    #else:
    #    image.set_shape(shape)

    #min_after_dequeue = 2 * batch_size
    #capacity = min_after_dequeue + 3 * batch_size

    #queue = tf.train.shuffle_batch(
    #    [image], batch_size=batch_size,
    #    num_threads=4, capacity=capacity,
    #    min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    #if dataset_name in ['celeba']:
    #    queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
    #    queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    #elif dataset_name in ['birds']:
    #    queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    #else:
    #    queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    #if data_format == 'NCHW':
    #    queue = tf.transpose(queue, [0, 3, 1, 2])
    #elif data_format == 'NHWC':
    #    pass
    #else:
    #    raise Exception("[!] Unkown data_format: {}".format(data_format))

    #return tf.to_float(queue)


def crop_and_resize(img, new_dim):
    """Takes square image object, center crops, and resizes to orig size."""
    width, height = img.size
    assert width == height, 'width must equal height'

    left = (width - new_dim)/2
    top = (height - new_dim)/2
    right = (width + new_dim)/2
    bottom = (height + new_dim)/2

    img = img.crop((left, top, right, bottom))
    img = img.resize((width, height), Image.BILINEAR)
    return img
