import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import cv2
import augmentations


class SegmentationDataGenerator(keras.utils.Sequence):
    'Generates data for segmentation (test)'

    def __init__(self, img_dir, mask_dir, n_classes, batch_size=4, input_dimensions=(1024, 2048), depth_dir=None, use_data_augmentation=False, shuffle_seed=None):
        """Initializes
        :param img_dir: path of the directory that contains the RGB images
        :param mask_dir: path of the directory that contains the mask images (labels)
        :param n_classes: number of classes
        :param batch_size: batch size for the training
        :param input_dimensions: dimension (H x W) required for the model's input
        :param depth_dir: path of the directory that contains the depth images (for RGB-D mode only). Default value is None, and it indicates that there is no depth channel in the dataset
        :param use_data_augmentation: flag that indicates whether data augmentation is used
        :param shuffle_seed: seed used for the random module to pseudo-randomly shuffle the dataset
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.use_data_augmentation = use_data_augmentation
        self.shuffle_seed = shuffle_seed
        self.depth_dir = depth_dir
        self.input_dimensions = input_dimensions

        random.seed(shuffle_seed)
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
            return X, Y
        else:
            return X, Z, Y

    def _read_image(self, image_path):
        """Reads the image from the image directory and returns the associated numpy array
        """
        img = cv2.imread(os.path.join(
            self.img_dir, image_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts to RGB
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
    datagen = SegmentationDataGenerator(
        'datasets/sun_rgbd/SUNRGBD-train_images',
        'datasets/sun_rgbd/train_labels',
        38,
        batch_size=3,
        input_dimensions=(530, 730),
        depth_dir='datasets/sun_rgbd/sunrgbd_train_depth',
        use_data_augmentation=True,
        shuffle_seed=9
    )
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
            
