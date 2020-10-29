# categorical focal loss function implemented with the help of  https://github.com/qubvel/segmentation_models/blob/94f624b7029deb463c859efbd92fa26f512b52b8/segmentation_models/base/functional.py#L259
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class CategoricalFocalLoss(keras.losses.Loss):
    """Implementation of categorical focal loss. Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    """

    def __init__(self, name=None, gamma=2.0, alpha=0.25):
        """
        :param name: displayed name for loss function
        :param gamma: gamma constant used in focal loss. gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more focus on hard misclassified example
        :param alpha: alpha constant used in focal loss equation. scalar factor to reduce the relative loss.
        """
        super(CategoricalFocalLoss, self).__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # # clip to prevent NaN's and Inf's
        # y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # # Calculate focal loss
        # loss = - y_true * (self.alpha * K.pow((1 - y_pred),
        #                                       self.gamma) * K.log(y_pred))
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))


class BinaryFocalLoss(keras.losses.Loss):
    """Implementation of simple binary focal loss.
    """

    def __init__(self, name=None, gamma=2.0, alpha=0.25):
        """
        :param name: displayed name for loss function
        :param gamma: gamma constant used in focal loss. gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more focus on hard misclassified example
        :param alpha: alpha constant used in focal loss equation. scalar factor to reduce the relative loss.
        """
        super(BinaryFocalLoss, self).__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        """

        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * self.alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        
        # Calculate cross entropy
        cross_entropy = - K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), self.gamma)
        
        # Calculate focal loss
        loss = weight * cross_entropy

        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss
