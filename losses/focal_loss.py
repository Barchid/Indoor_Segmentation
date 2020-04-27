# categorical focal loss function implemented with the help of  https://github.com/qubvel/segmentation_models/blob/94f624b7029deb463c859efbd92fa26f512b52b8/segmentation_models/base/functional.py#L259
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import losses_utils


class CategoricalFocalLoss(Loss):
    """Implementation of categorical focal loss. Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    """

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None, gamma=2.0, alpha=0.25):
        super(CategoricalFocalLoss, self).__init__(
            reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate focal loss
        loss = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))

        return K.mean(loss)
