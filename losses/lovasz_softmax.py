"""
All implementations are an update of the original code from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py
from tensorflow 1.x to tensorflow 2.x. There is also an adaptation of Lovasz-softmax multi-class loss in keras framework.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np


class MultiClassLovaszSoftmaxLoss(keras.losses.Loss):
    """
    Keras class for multi-class lovasz softmax loss.
    """

    def __init__(self, name=None, classes='present', per_image=False, ignore=None, order='BHWC'):
        """
        :param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        :param per_image: compute the loss per image instead of per batch
        :param ignore: void class labels
        :param order: use BHWC or BCHW
        """
        super(MultiClassLovaszSoftmaxLoss, self).__init__(name=name)
        self.classes = classes
        self.per_image = per_image
        self.ignore = self.ignore
        self.order = self.order

    def call(self, y_true, y_pred):
        # Reshape y_true of shape (B,H,W,C) to a tensor of ground truth labels (B,H,W)
        y_true = K.argmax(y_true)

        return lovasz_softmax(y_pred, y_true, classes=self.classes, per_image=self.per_image, ignore=self.ignore, order=self.order)


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.math.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        # foreground for class c
        fg = tf.cast(tf.math.equal(labels, c), probas.dtype)
        if classes == 'present':
            present.append(tf.math.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.math.abs(fg - class_pred)
        errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(
            errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(
                grad), 1, name="loss_class_{}".format(c))
        )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.math.reduce_mean(losses_tensor)
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.math.reduce_sum(gt_sorted)
    intersection = gts - tf.math.cumsum(gt_sorted)
    union = gts + tf.math.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard
