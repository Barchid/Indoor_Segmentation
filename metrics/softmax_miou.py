"""
Overrides the tf.keras.metrics.MeanIoU in order to use the one-hot encoded predictions instead of argmax.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class SoftmaxMeanIoU(keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None):
        super(SoftmaxMeanIoU, self).__init__(
            num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Change the softmax masks into the argmax equivalent to make the metric work
        y_true = K.argmax(y_true)
        y_pred = K.argmax(y_pred)

        return super(SoftmaxMeanIoU, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return super(SoftmaxMeanIoU, self).result()

    def reset_states(self):
        super(SoftmaxMeanIoU, self).reset_states()

    def get_config(self):
        return super(SoftmaxMeanIoU, self).get_config()
