import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import random
import cv2
import data_generators.augmentations as augmentations
import math


class ScenenetRGBDDataGenerator(keras.utils.Sequence):
    'Generates data for depth estimation (test)'

    def __init__(self, config, is_training_set=True):
        """Initializes the data generator used in the depth estimation task task.
        """
        # IF [the generator uses the training set]
        if is_training_set:
            self.img_dir = config.generator.img_dir
            self.depth_dir = config.generator.depth_dir
            self.mask_dir = config.generator.mask_dir
            self.use_data_augmentation = config.generator.use_data_augmentation

        # ELSE [means the generator uses the validation set]
        else:
            self.img_dir = config.validation.img_dir
            self.depth_dir = config.validation.depth_dir
            self.mask_dir = config.validation.mask_dir
            self.use_data_augmentation = False

        self.batch_size = config.trainer.batch_size
        self.shuffle_seed = config.generator.shuffle_seed
        self.input_dimensions = (config.model.height, config.model.width)
        self.config = config
        self.augmenter = augmentations.init_augmenter()
        self.n_classes = config.model.classes

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

        for directory, filename in batch:
            img_path = os.path.join(
                directory, 'photo', filename + ".jpg").replace('\\\\', '\\')
            mask_path = os.path.join(
                directory, 'class40', filename + ".png").replace('\\\\', '\\')
            depth_path = os.path.join(
                directory, 'depth', filename + ".png").replace('\\\\', '\\')

            img = self._read_image(img_path)
            mask = self._read_mask(mask_path)
            depth = self._read_depth(depth_path)

            # Launch data_augmentation if needed
            if self.use_data_augmentation:
                img, mask, depth = augmentations.process_augmentation_segmentation(
                    self.augmenter, img, mask, depth)

            X.append(self._get_image_tensor(img))
            Y.append(self._get_mask_tensor(mask))
            Z.append(self._get_depth_tensor(depth))

            # create the ground truth of data generator
            Y_array = np.array(Y)
            Z_array = np.array(Z)
            ground_truth = [Y_array]
            s = self.config.hdaf.s if isinstance(
                self.config.hdaf.s, int) else 1

            # add Y_array for each auxiliary seg output (one for each hdaf module but the last one)
            for i in range(s-1):
                ground_truth.append(Y_array)

            # add Z_array for each auxiliary depth output
            if self.config.hdaf.use_depth:
                for i in range(s):
                    ground_truth.append(Z_array)

        return np.array(X), ground_truth

    def _read_image(self, image_path):
        """Reads the image from the image directory and returns the associated numpy array
        """
        img = None
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts to RGB

        # Resize image
        img = cv2.resize(
            img, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)
        return img

    def _read_mask(self, mask_path):
        """Reads the mask image from the mask directory and returns the associated numpy array
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize image
        mask = cv2.resize(
            mask, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)

        return mask

    def _read_depth(self, depth_path):
        """Reads the depth image from the depth directory and returns the associated numpy array
        """
        depth = np.array(Image.open(depth_path))
        # normalize between 0 & 1
        depth = ((depth - np.min(depth)) / (np.max(depth) - np.min(depth)))

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

    def _get_depth_tensor(self, depth):
        """Formats the depth image in parameter as a tensor that can be used by the network.
        """
        depth = depth.astype(np.float32)
        # depth = depth/255.  # normalize between 0 and 1

        # expand dimension if depth tensor has only 2 dims (and not 3)
        if depth.ndim == 2:
            depth = np.expand_dims(depth, 2)
        return depth

    def _get_mask_tensor(self, raw_mask):
        """Formats the raw mask in parameter to a tensor that can be used by the network.
        The mask will be a tensor H x W x n_classes.
        """
        mask = np.zeros((raw_mask.shape[0], raw_mask.shape[1], self.n_classes))

        # put 1 where the pixel of the mask belongs to the focused channel (representing a class to segment)
        for c in range(self.n_classes):
            mask[:, :, c] = (raw_mask == c).astype(int)

        return mask

    def _get_data_tuples(self):
        'Returns a list of tuples with directory path + filename'
        result = []

        # generate directory + filenames
        for part in self.config.generator.parts:
            part = os.path.normpath(os.path.join(self.img_dir, part))
            # remove double backslashes for windows
            part = part.replace('\\\\', '\\')

            for directory in os.listdir(part):
                directory = os.path.normpath(os.path.join(part, directory))
                directory = directory.replace('\\\\', '\\')

                for i in range(0, 300):
                    # use the picture in the dataset according to the configuration's step (ex : 1/10 picture)
                    if i % self.config.generator.img_step == 0:
                        pic_num = i * 25
                        result.append((directory, str(pic_num)))

        # shuffles the dataset
        random.shuffle(result)

        return result
