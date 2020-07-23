from utils import factory
from data_generators.joint_data_generator import JointDataGenerator
from utils.config import process_config
import cv2
import numpy as np
from utils.utils import get_args
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from losses.focal_tversky_loss import class_tversky, focal_tversky_loss


def webcam(model, config, datagen):
    """Visualization of predictions
    """

    for i in range(len(datagen)):
        print('Processing sample n°', i, '...')
        # IF there is no depth data
        X, gts = datagen[i]
        gt_seg = gts[0]
        gt_dep = gts[-1]

        preds = model.predict(X, batch_size=1, verbose=1)
        pred_seg = preds[0]
        pred_dep = preds[-1]
        
        pred_dep = np.clip(pred_dep[0], 0., 1.)
        pred_seg = pred_seg[0]


def main():
        # capture the config path from the run arguments
        # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Force some parameters from configuration
    config.generator.use_data_augmentation = False
    config.trainer.batch_size = 1

    # Data generator creation
    datagen = JointDataGenerator(config, is_training_set=False)

    network = factory.create(config.model.class_name)(config, datagen)
    model = network.model

    for i in range(len(datagen)):
        print('Processing sample n°', i, '...')
        # IF there is no depth data
        X, gts = datagen[i]
        gt_seg = gts[0]
        gt_dep = gts[-1]

        preds = model.predict(X, batch_size=1, verbose=1)
        pred_seg = preds[0]
        pred_dep = preds[-1]

        pred_dep = np.clip(pred_dep[0], 0., 1.)
        pred_seg = pred_seg[0]


if __name__ == "__main__":
    main()
