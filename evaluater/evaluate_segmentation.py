"""Evaluates the segmentation task
"""


def evaluate(model, data_generator):
    """Evaluates the pixel accuracy, mean pixel accuracy, per-class IoUs and general mIoU of a trained network
    :param model: the trained Keras model
    :param data_generator: the instance of SegmentationDataGenerator that generates the test set
    """
    