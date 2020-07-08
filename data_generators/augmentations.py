# Contains the transformations to use in the data augmentation process
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import cv2
import numpy as np


def sometimes(augmenter):
    """Applies a "Sometimes" meta augmenter from imgaug to the specified augmenter in parameter
    """
    return iaa.Sometimes(0.5, augmenter)


def init_augmenter(img_mode="color"):
    """Initializes the augmenters used in the training dataset
    :param config: the config object that contains all the 
    """
    ia.seed(10)

    if img_mode == 'color':
        return iaa.Sequential([
            sometimes(iaa.Fliplr()),
            iaa.MultiplyBrightness((0.6, 1.4)),
            # TODO: try no ChangeColor or Brightness
            sometimes(iaa.ChangeColorTemperature((5000, 7000))),
            iaa.Crop(percent=(
                (0, 0.50),
                (0, 0.50),
                (0, 0.50),
                (0, 0.50)
            )),
            sometimes(iaa.OneOf([
                iaa.Cutout(nb_iterations=(1, 4), size=0.2,
                           squared=False, cval=(0, 255), fill_mode="constant"),
                iaa.Cutout(nb_iterations=(1, 4), size=0.2, squared=False, cval=(
                    0, 255), fill_mode="gaussian", fill_per_channel=True),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
            ]))
        ])
    else:
        return iaa.Sequential([
            sometimes(iaa.Fliplr()),
            iaa.Crop(percent=(
                (0, 0.40),
                (0, 0.40),
                (0, 0.40),
                (0, 0.40)
            ))
        ])


def process_augmentation_segmentation(augmenter, image, mask, depth=None):
    """
    processes data augmentation for the current image and the appropriate mask for semantic segmentation.
    """
    segmap = SegmentationMapsOnImage(mask, shape=image.shape)

    # use depth image
    if not depth is None:
        # convert depth into heatmap to use it in imgaug
        heatmap = HeatmapsOnImage(depth.astype(
            np.float32), shape=image.shape, min_value=0.0, max_value=255.0)
        aug_image, aug_mask, aug_depth = augmenter(
            image=image, segmentation_maps=segmap, heatmaps=heatmap)
        return aug_image, aug_mask.get_arr(), aug_depth.get_arr().astype(np.uint8)
    # only RGB
    else:
        aug_image, aug_seg = augmenter(image=image, segmentation_maps=segmap)
        return aug_image, aug_seg.get_arr(), None


def process_augmentation_depth_estimation(augmenter, image, depth):
    """
    processes data augmentation for the current image and the appropriate mask 
    """
    # convert depth into heatmap to use it in imgaug
    heatmap = HeatmapsOnImage(depth.astype(
        np.float32), shape=image.shape, min_value=0.0, max_value=255.0)
    aug_image, aug_depth = augmenter(
        image=image, heatmaps=heatmap)
    return aug_image, aug_depth.get_arr().astype(np.uint8)


if __name__ == "__main__":
    # tests
    image = cv2.imread(
        'datasets/sun_rgbd/SUNRGBD-test_images/1.jpg', flags=cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(
        'datasets/sun_rgbd/test_labels/1.png', cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(
        'datasets/sun_rgbd/sunrgbd_test_depth/1.png', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('', image)
    cv2.waitKey()
    cv2.imshow('', mask)
    cv2.waitKey()
    cv2.imshow('', depth)
    cv2.waitKey()

    aug_img, aug_seg, aug_depth = process_augmentation_segmentation(
        image, mask)
    from tensorflow import keras
    print(keras.backend.one_hot(aug_seg, 38))
    exit()
    cv2.imshow('', aug_img)
    cv2.waitKey(0)
    cv2.imshow('', aug_seg)
    cv2.waitKey(0)
    cv2.imshow('', aug_depth)
    cv2.waitKey()

    print(aug_img.dtype, aug_seg.dtype, aug_depth.dtype)
