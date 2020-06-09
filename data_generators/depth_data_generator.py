import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import cv2
import data_generators.augmentations as augmentations
import math


class DepthDataGenerator(keras.utils.Sequence):
    'Generates data for depth estimation (test)'

    def __init__(self, config, is_training_set=True):
        """Initializes the data generator used in the depth estimation task task.
        """
        # IF [the generator uses the training set]
        if is_training_set:
            self.img_dir = config.generator.img_dir
            self.depth_dir = config.generator.depth_dir
            self.use_data_augmentation = config.generator.use_data_augmentation

        # ELSE [means the generator uses the validation set]
        else:
            self.img_dir = config.validation.img_dir
            self.depth_dir = config.generator.depth_dir
            self.use_data_augmentation = False

        self.batch_size = config.trainer.batch_size
        self.n_classes = config.model.classes
        self.shuffle_seed = config.generator.shuffle_seed
        self.input_dimensions = (config.model.height, config.model.width)
        self.config = config
        self.augmenter = augmentations.init_augmenter(
            img_mode=self.config.generator.img_mode)

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
        X, Y = [], []
        batch = self.data_tuples[index *
                                 self.batch_size: (index+1) * self.batch_size]

        for img_path, depth_path in batch:
            img = self._read_image(img_path)
            depth = self._read_depth(depth_path)

            # Launch data_augmentation if needed
            if self.use_data_augmentation:
                img, depth = augmentations.process_augmentation_depth_estimation(
                    self.augmenter, img, depth)

            X.append(self._get_image_tensor(img))
            Y.append(self._get_depth_tensor(depth))

        return np.array(X), np.array(Y)

    def _read_image(self, image_path):
        """Reads the image from the image directory and returns the associated numpy array
        """
        img = None
        # COLOR mode
        if self.config.generator.img_mode == 'color':
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts to RGB
        # GRAYSCALE mode
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Resize image
        img = cv2.resize(
            img, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)
        return img

    def _read_depth(self, depth_path):
        """Reads the depth image from the depth directory and returns the associated numpy array
        """
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
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

        if self.config.generator.img_mode == 'grayscale':
            img = np.expand_dims(img, 2)

        return img

    def _get_depth_tensor(self, depth):
        """Formats the depth image in parameter as a tensor that can be used by the network.
        """
        depth = depth.astype(np.float32)
        depth = depth/255.  # normalize between 0 and 1
        return depth

    def _get_data_tuples(self):
        'Returns a list of tuples that contain the image path and related depth path'
        result = []

        for img_file in os.listdir(self.img_dir):
            img_path = os.path.join(self.img_dir, img_file)

            # check if img_path is a file
            if not os.path.isfile(img_path):
                continue

            data_id = os.path.splitext(img_file)[0]
            depth_file = data_id + \
                ".png" if os.path.exists(os.path.join(
                    self.depth_dir, data_id + '.png')) else data_id + ".jpg"
            depth_path = os.path.join(self.depth_dir, depth_file)
            result.append((img_path, depth_path))

        # shuffles the dataset
        random.shuffle(result)

        return result
