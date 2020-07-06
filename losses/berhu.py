# tensorflow Keras implementation of berHu loss for monocular depth estimation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class BerHuLoss(keras.losses.Loss):
    """Implementation of berHu loss for depth estimation.
    """

    def call(self, y_true, y_pred):
        y_true, y_pred = K.batch_flatten(y_true), K.batch_flatten(y_pred)
        col_shape = K.shape(y_true)[1]

        error = y_pred - y_true
        abs_error = K.abs(error)

        # delta is chosen here
        delta = 0.2 * K.max(abs_error, axis=1)
        delta = K.expand_dims(delta)  # delta is now (batch, 1)
        # delta is the same shape a y_pred/y_true
        delta = tf.tile(delta, [1, col_shape])

        # small error is abs_error
        # big error formula is :
        big_error = K.square(error) + K.square(delta)
        big_error = tf.divide(big_error, 2 * delta)

        berhu = tf.where(tf.greater(error, delta), big_error, abs_error)
        return tf.reduce_sum(berhu)
