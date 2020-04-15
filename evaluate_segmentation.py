"""Evaluates the segmentation task. Takes the test dataset paths in parameter and create the class-specific mIoUs, the general mIoU and the accuracy.
"""
import argparse
from utils.utils import get_args
from utils.config import process_config
from data_generators.segmentation_data_generator import SegmentationDataGenerator
import numpy as np
import cv2
from models.fast_scnn_nyuv2 import FastScnnNyuv2

# imports for visualization
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def evaluate(model, config):
    """Evaluates the performances from the trained model in parameter.
    :param model: the trained Keras model
    :param config: the config object created from JSON configuration
    """
    # Data generator creation
    datagen = SegmentationDataGenerator(config, is_training_set=False)

    for i in range(len(datagen)):
        # IF there is depth data
        if config.generator.depth_dir is None:
            X, Y = datagen[i]
            prediction = model.predict(
                X, batch_size=config.trainer.batch_size, verbose=1)
            prediction = np.argmax(prediction)
            print(prediction.shape)
        else:
            # TODO : depth dir inclusion
            pass


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


def visualize_results(model, config):
    """Shows the predictions from the trained model in parameter.
    :param model: the trained Keras model
    :param config: the config object created from JSON configuration
    """

    # Data generator creation
    datagen = SegmentationDataGenerator(config, is_training_set=False)

    for i in range(len(datagen)):
        # IF there is depth data
        if config.generator.depth_dir is None:
            X, Y = datagen[i]
            prediction = model.predict(
                X, batch_size=config.trainer.batch_size, verbose=1)

            visualization = create_visualization(
                X[0], prediction[0], Y[0])

            cv2.imshow('Image - Ground truth - Prediction', visualization)
            if cv2.waitKey() == ord('a'):
                print('STOP')
                exit()

        else:
            # TODO : depth dir inclusion
            pass


def load_weights(model, config):
    """Load weights file if required by configuration
    """
    if not hasattr(config, 'evaluation') or not hasattr(config.evaluation, 'weights_file'):
        return

    print('Load weight file : ', config.evaluation.weights_file)
    model.load_weights(config.evaluation.weights_file)


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args(test_args=True)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Force some parameters from configuration
    config.generator.use_data_augmentation = False
    config.trainer.batch_size = 1

    print("Instantiate the model")
    network = FastScnnNyuv2(config)

    print("Loading weights")
    load_weights(network.model, config)

    if args.visualize:
        print("Beginning visualization")
        visualize_results(network.model, config)
    else:
        print('Beginning evaluation')
        evaluate(network.model, config)


if __name__ == "__main__":
    main()
