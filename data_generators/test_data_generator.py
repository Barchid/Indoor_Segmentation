import tensorflow as tf
import tensorflow.keras
import numpy as np

class TestDataGenerator(keras.utils.Sequence):
    'Generates data for segmentation (test)'

    def __init__(self, img_path, mask_path, n_classes, batch_size=4, use_data_augmentation=False, shuffle_seed=None):
        """Initializes
        :param img_path: path of the directory that contains the RGB images
        :param mask_path: path of the directory that contains the mask images (labels)
        :param n_classes: number of classes
        :param batch_size: batch size for the training
        :param use_data_augmentation: flag that indicates whether data augmentation is used
        :param shuffle_seed: seed used for the random module to pseudo-randomly shuffle the dataset
        """
        self.train_img_path = train_img_path
        self.train_mask_path = train_mask_path
        self.test_img_path = test_img_path
        self.test_mask_path = test_mask_path
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.use_data_augmentation = use_data_augmentation
        self.shuffle_seed = shuffle_seed

        # list img directory
        
        # list mask directory

        pass

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        pass

    def _get_data_tuples(self):
        pass