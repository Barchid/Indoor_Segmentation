"""Proceeds to a qualitative evaluation = creates visualization of the predicted segmentation map.
"""
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np


def create_visualization(image, y_pred, y_true):
    """Creates the qualitative results to visualize
    """
    # format prediction and ground truth
    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)
    y_true = np.argmax(y_true, axis=-1).astype(np.uint8)

    # format raw image
    # put values between 0 and 255 (before it was between 0 and 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image * 255.
    image = image.astype(np.uint8)

    # generate segmaps
    y_pred = SegmentationMapsOnImage(y_pred, shape=image.shape)
    y_true = SegmentationMapsOnImage(y_true, shape=image.shape)

    grid_image = ia.draw_grid([
        image,
        y_true.draw_on_image(image)[0],
        y_pred.draw_on_image(image)[0]
    ], cols=3)

    return grid_image


def visualize_results(model, config, datagen):
    """Visualization of predictions
    """

    for i in range(len(datagen)):
        print('Processing sample n°', i, '...')
        # IF there is no depth data
        if config.generator.depth_dir is None:
            X, Y = datagen[i]
            prediction = model.predict(
                X, batch_size=config.trainer.batch_size, verbose=1)

            visualization = create_visualization(X[0], prediction[0], Y[0])

            cv2.imshow('Image - Ground truth - Prediction', visualization)
            if cv2.waitKey() == ord('a'):
                print('STOP')
                exit()

        else:
            # TODO : depth dir
            pass
