"""Utilities Lambda layers"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


def resize_img(x, H, W, name=None):
    """Resizes the input image tensor with heigh H and width W
    """
    return Lambda(lambda x: tf.image.resize(x, (H, W)), name=name)(x)


def segmentation_head(features, classes, height, width, id="0"):
    features = conv2d(features, 64, 1, 1, kernel_size=3,
                      name="segmentation_head_3x3")
    features = conv2d(features, classes,
                      1, 1, kernel_size=1, name="segmentation_head_1x1")

    upsampled = resize_img(
        features, height, width)

    segmentation_mask = Activation(
        'softmax', name='seg_out')(upsampled)
    return segmentation_mask


def depth_head(features, classes, height, width, id="0"):
    features = conv2d(features, 64, 1, 1, kernel_size=3,
                      name="depth_head_" + id + "_3x3")
    features = conv2d(features, 1, 1, 1, kernel_size=1,
                      name="depth_head_" + id + "_1x1")

    upsampled = resize_img(
        features, self.config.model.height, self.config.model.width, name="dep_out_" + id)

    return upsampled


def conv2d(input, filters, stride, n, kernel_size=3, name=None):
    x = input
    for i in range(n):
        # define names for layers
        if name is not None:
            conv_name = name + "_conv_" + str(i)
            bn_name = name + "_bn_" + str(i)
            swish_name = name + "_swish_" + str(i)
        else:
            conv_name = bn_name = swish_name = None

        x = Conv2D(filters, (kernel_size, kernel_size), strides=(
            stride, stride), padding="same", name=conv_name)(x)

        x = BatchNormalization(name=bn_name)(x)

        x = Activation(lambda a: tf.nn.swish(a), name=swish_name)(x)
    return x
