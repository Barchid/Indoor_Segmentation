import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import cv2
import data_generators.augmentations as augmentations
import math


class SegmentationDataGenerator(keras.utils.Sequence):
    'Generates data for segmentation (test)'

    def __init__(self, config, is_training_set=True):
        """Initializes the data generator used in the segmentation task.
        """
        # IF [the generator uses the training set]
        if is_training_set:
            self.img_dir = config.generator.img_dir
            self.mask_dir = config.generator.mask_dir
            self.depth_dir = None if not hasattr(
                config.generator, 'depth_dir') else config.generator.depth_dir
            self.use_data_augmentation = config.generator.use_data_augmentation

        # ELSE [means the generator uses the validation set]
        else:
            self.img_dir = config.validation.img_dir
            self.mask_dir = config.validation.mask_dir
            self.depth_dir = None if not hasattr(
                config.validation, 'depth_dir') else config.generator.depth_dir
            self.use_data_augmentation = False

        self.batch_size = config.trainer.batch_size
        self.n_classes = config.model.classes
        self.shuffle_seed = config.generator.shuffle_seed
        self.input_dimensions = (config.model.height, config.model.width)
        self.config = config

        random.seed(self.shuffle_seed)
        self.data_tuples = self._get_data_tuples()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.data_tuples) / self.batch_size)
        # return int(np.floor(len(self.data_tuples) / self.batch_size))

    def on_epoch_end(self):
        'shuffles the data tuples after each epoch'
        random.shuffle(self.data_tuples)

    def __getitem__(self, index):
        # Retrieve the paths used in the current batch
        X, Y, Z = [], [], []
        batch = self.data_tuples[index *
                                 self.batch_size: (index+1) * self.batch_size]

        for (img_path, mask_path) in batch:
            img = self._read_image(img_path)
            mask = self._read_mask(mask_path)

            # IF [depth mode is enabled]
            if not self.depth_dir is None:
                # depth_path and mask_path have same filename
                depth = self._read_depth(mask_path)
            else:
                depth = None

            # Launch data_augmentation if needed
            if self.use_data_augmentation:
                img, mask, depth = augmentations.process_augmentation(
                    img, mask, depth=depth)

            X.append(self._get_image_tensor(img))
            Y.append(self._get_mask_tensor(mask))
            if not self.depth_dir is None:  # add augmented depth if available
                Z.append(self._get_depth_tensor(depth))

        if self.depth_dir is None:  # No depth available
            return np.array(X), np.array(Y)
        else:
            return X, Z, Y

    def _read_image(self, image_path):
        """Reads the image from the image directory and returns the associated numpy array
        """
        img = None
        # COLOR mode
        if self.config.generator.img_mode == 'color':
            img = cv2.imread(os.path.join(
                self.img_dir, image_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts to RGB
        # GRAYSCALE mode
        else:
            img = cv2.imread(os.path.join(
                self.img_dir, image_path), cv2.IMREAD_GRAYSCALE)
        # Resize image
        img = cv2.resize(
            img, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)
        return img

    def _read_mask(self, mask_path):
        """Reads the mask image from the mask directory and returns the associated numpy array
        """
        mask = cv2.imread(os.path.join(
            self.mask_dir, mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize image
        mask = cv2.resize(
            mask, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)

        return mask

    def _read_depth(self, depth_path):
        """Reads the depth image from the depth directory and returns the associated numpy array
        """
        # Choice between ".jpg" or ".png" extension in the depth images
        data_id = os.path.splitext(depth_path)[0]
        depth_path = data_id + \
            ".png" if os.path.exists(os.path.join(
                self.depth_dir, data_id + '.png')) else data_id + ".jpg"

        depth = cv2.imread(os.path.join(
            self.depth_dir, depth_path), cv2.IMREAD_GRAYSCALE)
        # Resize image
        depth = cv2.resize(
            depth, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)

        return depth

    def _get_image_tensor(self, img):
        """returns a tensor to use in the network from the img in parameter
        :param img: the image to format
        """
        img = img.astype(np.float32)
        img = img/255.  # normalize between 0 and 1
        return img

    def _get_mask_tensor(self, raw_mask):
        """Formats the raw mask in parameter to a tensor that can be used by the network.
        The mask will be a tensor H x W x n_classes.
        """
        mask = np.zeros((raw_mask.shape[0], raw_mask.shape[1], self.n_classes))

        # put 1 where the pixel of the mask belongs to the focused channel (representing a class to segment)
        for c in range(self.n_classes):
            mask[:, :, c] = (raw_mask == c).astype(int)

        return mask

    def _get_depth_tensor(self, depth):
        """Formats the depth image in parameter as a tensor that can be used by the network.
        """
        depth = depth.astype(np.float32)
        depth = depth/255.  # normalize between 0 and 1
        return depth

    def _get_data_tuples(self):
        'Returns a list of tuples that contain the image path and related mask path'
        result = []

        for img_path in os.listdir(self.img_dir):
            # check if img_path is a file
            if not os.path.isfile(os.path.join(self.img_dir, img_path)):
                continue

            data_id = os.path.splitext(img_path)[0]
            mask_path = data_id + '.png'
            result.append((img_path, mask_path))

        # shuffles the dataset
        random.shuffle(result)

        return result


if __name__ == "__main__":
    # TEST the implementation
    from dotmap import DotMap
    config = {
        'generator': {
            "img_dir": "datasets/sun_rgbd/SUNRGBD-train_images",
            "mask_dir": "datasets/sun_rgbd/train_labels",
            "depth_dir": 'datasets/sun_rgbd/sunrgbd_train_depth',
            "use_data_augmentation": True,
            "shuffle_seed": 9
        },
        "model": {
            "optimizer": "SGD",
            "learning_rate": 0.045,
            "momentum": 0.9,
            "width": 320,
            "height": 320,
            "classes": 38
        },
        "trainer": {
            "num_epochs": 20,
            "batch_size": 8,
            "verbose_training": True,
            "workers": 2
        },
    }
    config = DotMap(config)
    datagen = SegmentationDataGenerator(config)
    print(len(datagen.data_tuples))
    print(datagen.data_tuples[0])

    def showim(im):
        cv2.imshow('im', im)
        if cv2.waitKey() == ord('a'):
            print('STOP')
            exit()

    for i in range(len(datagen)):
        X, Z, Y = datagen[i]
        for img, depth, mask in zip(X, Z, Y):
            print(img.shape, depth.shape, mask.shape)
            showim(img)
            showim(depth)
