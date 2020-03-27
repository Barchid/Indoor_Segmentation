import tensorflow as tf
# import tensorflow.keras
import keras
from keras import backend as K


class SegmentationMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, label=None, name="Mean IoU", **kwargs):
        """Computes the mean intersection over union metric for the segmentation task.
        This metric can compute the mIoU for all the classes or one class in particular.

        :param num_classes: the number of classes of the segmentation task
        :param label: the integer that represents the class for which the mIoU is computed. If label=None, the mIoU is computed for all the classes.
        :param name: the name to display the measure
        :param **kwargs
        """
        super(SegmentationMeanIoU).__init__(name=name, **kwargs)
        self.label = label
        self.num_classes = num_classes

        # The number of predictions that was done during the epoch
        self.pred_count = self.add_weight(
            name='pred_count',
            initializer='zeros'
        )

        # the IoU summation of all the predictions done during the epoch.
        self.iou_sum = self.add_weight(
            name='iou_sum',
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the pred_count and iou_sum by calculating all the IoUs for all the images presented in batch

        :param y_true: the batch of ground truth masks. The shape is (B, H, W, C) where B is batch size, H is height, W is width and C is the number of classes.
        :param y_pred: the batch of segmentations predicted by the model. Shape is same as y_true.
        :param sample_weight: not implemented here.
        """
        pass

    def result(self):
        """Called at the end of the epoch.
        """
        pass

    def reset_states(self):
        pass
