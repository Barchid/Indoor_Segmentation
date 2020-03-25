import tensorflow as tf
import tensorflow.keras
import numpy as np
import os
import random
import cv2


class SegmentationDataGenerator(keras.utils.Sequence):
    'Generates data for segmentation (test)'

    def __init__(self, img_dir, mask_dir, n_classes, batch_size=4, input_dimensions=(1024, 2048), depth_dir=None, use_data_augmentation=False, shuffle_seed=None, prediction_mode=False):
        """Initializes
        :param img_dir: path of the directory that contains the RGB images
        :param mask_dir: path of the directory that contains the mask images (labels)
        :param n_classes: number of classes
        :param batch_size: batch size for the training
        :param input_dimensions: dimension (H x W) required for the model's input
        :param depth_dir: path of the directory that contains the depth images (for RGB-D mode only). Default value is None, and it indicates that there is no depth channel in the dataset
        :param use_data_augmentation: flag that indicates whether data augmentation is used
        :param shuffle_seed: seed used for the random module to pseudo-randomly shuffle the dataset
        :param prediction_mode: flag that indicates whether the generator returns also the mask data (training/testing) or if it is only used for prediction (no mask labels)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.use_data_augmentation = use_data_augmentation
        self.shuffle_seed = shuffle_seed
        self.prediction_mode = prediction_mode
        self.depth_dir = depth_dir
        self.input_dimensions = input_dimensions

        random.seed(self.shuffle_seed)
        self.data_tuples = self._get_data_tuples()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_tuples) / self.batch_size))

    def __getitem__(self, index):
        # Retrieve the paths used in the current batch
        X, Y, Z = [], [], []
        batch = self.data_tuples[index *
                                 self.batch_size: (index+1) * self.batch_size]

        for (img_path, mask_path) in batch:
            img = self._get_image_tensor(img_path)
            X.append(img)

            if not self.prediction_mode:
                mask = self._get_mask_tensor(mask_path)
                Y.append(mask)

            # IF [depth mode is enabled]
            if self.depth_dir != None:
                # depth_path and img_path have same filename
                depth = self._get_depth_tensor(img_path)
                Z.append(depth)

        if self.depth_dir == None:
            if self.prediction_mode:
                return X
            else:
                return X, Y
        else:
            if self.prediction_mode:
                return X, Z
            else:
                return X, Z, Y

    def _get_image_tensor(self, image_path):
        """Retrieves and format the image specified by the path in parameter and returns a tensor to use in the network
        :param image_path: the path to the file of the image
        """
        img = cv2.imread(os.path.join(
            self.img_dir, image_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts to RGB
        # Resize image
        img = cv2.resize(
            img, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)
        img = img/255.  # normalize between 0 and 1
        return img

    def _get_mask_tensor(self, mask_path):
        """Retrieves the mask image as a tensor from the specified path in parameter.
        The mask will be a tensor H x W x n_classes.
        """
        raw_mask = cv2.imread(os.path.join(
            self.mask_dir, mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize image
        raw_mask = cv2.resize(
            raw_mask, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros(raw_mask.shape[0], raw_mask.shape[1], self.n_classes)

        # put 1 where the pixel of the mask belongs to the focused channel (representing a class to segment)
        for c in range(self.n_classes):
            mask[:, :, c] = (raw_mask == c).astype(int)

        return mask

    def _get_depth_tensor(self, depth_path):
        """Retrieves the depth image as a tensor from the specified path in parameter.
        """
        depth = cv2.imread(os.path.join(
            self.depth_dir, depth_path), cv2.IMREAD_GRAYSCALE)
        # Resize image
        depth = cv2.resize(
            depth, (self.input_dimensions[1], self.input_dimensions[0]), interpolation=cv2.INTER_NEAREST)
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
