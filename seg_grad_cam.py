"""Evaluates the segmentation task. Takes the test dataset paths in parameter and create the class-specific mIoUs, the general mIoU and the accuracy.
"""
from data_generators.segmentation_data_generator import SegmentationDataGenerator
import argparse
from utils import factory
from utils.utils import get_args
from utils.config import process_config
from utils.seg_grad_cam import SegGradCam
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args(seg_grad_cam=True)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Force some parameters from configuration
    config.generator.use_data_augmentation = False
    config.trainer.batch_size = 1

    # Launch evaluation
    eval_seg_grad_cam(config, args)


def eval_seg_grad_cam(config, args):
    # data generator on validation set
    datagen = SegmentationDataGenerator(config, is_training_set=False)

    # get trained model
    model = setup_model(config, datagen)

    # iterate through images
    for i in range(len(datagen)):
        print('Processing sample n°', i, '...')
        # IF there is no depth data
        if config.generator.depth_dir is None:
            X, Y = datagen[i]
            prediction = model.predict(
                X, batch_size=config.trainer.batch_size, verbose=1)

            # get heatmap from SEG-GRAD-CAM
            gradcam = SegGradCam(
                model, class_id=int(args.class_id), layer_name=args.layer)
            heatmap = gradcam.compute_heatmap(X)
            heatmap, overlay = gradcam.overlay_heatmap(heatmap, X[0])

            visualization = create_visualization(
                X[0], prediction[0], Y[0], heatmap, overlay)

            cv2.imshow('Image - Ground truth - Prediction', visualization)
            if cv2.waitKey() == ord('a'):
                print('STOP')
                exit()

        else:
            # TODO : depth dir
            pass


def setup_model(config, datagen):
    """load model & load weights"""
    network = factory.create(config.model.class_name)(config, datagen)
    network.model.load_weights(config.validation.weights_file)
    return network.model


def create_visualization(image, y_pred, y_true, heatmap, overlay):
    """Creates the qualitative results to visualize
    """
    # format prediction and ground truth
    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)
    y_true = np.argmax(y_true, axis=-1).astype(np.uint8)

    y_true_gray = cv2.cvtColor(y_true, cv2.COLOR_GRAY2RGB)

    # format raw image
    # put values between 0 and 255 (before it was between 0 and 1)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # generate segmaps
    y_pred = SegmentationMapsOnImage(y_pred, shape=image.shape)
    y_true = SegmentationMapsOnImage(y_true, shape=image.shape)

    grid_image = ia.draw_grid([
        image,
        y_true.draw_on_image(image)[0],
        y_pred.draw_on_image(image)[0],
        heatmap,
        overlay,
        y_true_gray
    ], rows=2, cols=3)

    return grid_image


if __name__ == "__main__":
    main()
