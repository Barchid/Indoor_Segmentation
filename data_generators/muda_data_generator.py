import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import cv2
import data_generators.augmentations as augmentations
import math


class MudaDataGenerator(keras.utils.Sequence):
    'Generates data for segmentation (test)'

    def __init__(self, config, is_training_set=True):
        """Initializes the data generator used in the segmentation task.
        """
        # IF [the generator uses the training set]
        if is_training_set:
            self.img_dir = config.generator.img_dir
            self.mask_dir = config.generator.mask_dir
            self.use_data_augmentation = config.generator.use_data_augmentation

        # ELSE [means the generator uses the validation set]
        else:
            self.img_dir = config.validation.img_dir
            self.mask_dir = config.validation.mask_dir
            self.use_data_augmentation = False

        self.batch_size = config.trainer.batch_size
        self.n_classes = config.model.classes
        self.shuffle_seed = config.generator.shuffle_seed
        self.input_dimensions = (config.model.height, config.model.width)
        self.config = config
        # configuration of each class for the relative binary mask
        self.classes = config.classes
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

        for (img_path, mask_path) in batch:
            img = self._read_image(img_path)
            mask = self._read_mask(mask_path)

            # Launch data_augmentation if needed
            if self.use_data_augmentation:
                img, mask, _ = augmentations.process_augmentation_segmentation(
                    self.augmenter, img, mask)

            X.append(self._get_image_tensor(img))
            Y.append(self._get_mask_tensor(mask))

        # compute ground truth with the batch of seg masks in list Y
        Y = np.array(Y)
        bin_masks = self._get_binary_masks(Y)
        ground_truth = []
        ground_truth.extend(bin_masks)
        ground_truth.append(Y)

        return np.array(X), ground_truth

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

    def _get_image_tensor(self, img):
        """returns a tensor to use in the network from the img in parameter
        :param img: the image to format
        """
        img = img.astype(np.float32)
        img = img/255.  # normalize between 0 and 1

        if self.config.generator.img_mode == 'grayscale':
            img = np.expand_dims(img, 2)

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

    def _get_binary_masks(self, mask):
        """
        From the batch of segmentation map of dim (b,H,W,c) to a list of c binary masks of dim (b,H,W,1)
        """
        bin_masks = []
        for c in range(self.n_classes):
            bin_mask_batch = mask[:, :, :, c]

            if bin_mask_batch.ndim == 3:  # add last dimension
                bin_mask_batch = np.expand_dims(bin_mask_batch, 3)

            bin_masks.append(bin_mask_batch)

        return bin_masks

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
