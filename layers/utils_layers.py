"""Utilities Lambda layers"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


def resize_img(x, H, W):
    """Resizes the input image tensor with heigh H and width W
    """
    return Lambda(lambda x: tf.image.resize(x, (H, W)))(x)
