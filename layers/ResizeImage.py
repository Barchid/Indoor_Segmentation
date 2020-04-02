import tensorflow as tf
from tensorflow import keras


class ResizeImage(keras.layers.Layer):
    """Layer used to resize the image tensor in input to use the right dimensions.
    This is a kind of layer to use in the 
    """
    def __init__(self, height, width, **kwargs):
        """
        :param height: the new height of the image tensor input
        :param width: the new width of the image tensor input
        """
        super(ResizeImage, self).__init__(**kwargs)
        self.dimensions = height, width

    def call(self, inputs, mask=None):
        return tf.image.resize(inputs, self.dimensions)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions[0], self.dimensions[1], input_shape[-1]