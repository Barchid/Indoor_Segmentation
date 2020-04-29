"""
Overrides the tf.keras.metrics.MeanIoU in order to use the one-hot encoded predictions instead of argmax.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class SoftmaxSingleMeanIoU(keras.metrics.MeanIoU):
    """
    mIoU metric specific to only one class that uses softmax input (not argmax input).
    """
    def __init__(self, label, name=None, dtype=None):
        """
        :param name: name used for the metric
        :param label: ID (integer) of the class used
        """
        super(SoftmaxSingleMeanIoU, self).__init__(
            2, name=name, dtype=dtype)
        self.label = label

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Change the softmax masks into the argmax equivalent to make the metric work
        y_true = K.cast(K.equal(K.argmax(y_true), self.label), K.floatx())
        y_pred = K.cast(K.equal(K.argmax(y_pred), self.label), K.floatx())

        return super(SoftmaxSingleMeanIoU, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return super(SoftmaxSingleMeanIoU, self).result()

    def reset_states(self):
        super(SoftmaxSingleMeanIoU, self).reset_states()

    def get_config(self):
        return super(SoftmaxSingleMeanIoU, self).get_config()


class SoftmaxMeanIoU(keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None):
        """
        :param: num_classes: number of class available
        :param name: name used for the metric
        """
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
