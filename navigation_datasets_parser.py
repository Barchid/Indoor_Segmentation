import cv2
import numpy as np
import os


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def change_masks(mask):
    new_mask = np.full_like(mask, 4)  # class 5 = other objects
    new_mask[mask == 0] = 0  # void class
    new_mask[mask == 1] = 1  # wall class
    new_mask[mask == 2] = 2  # floor class
    new_mask[mask == 31] = 3  # person class
    return new_mask


def transfer_mask_dir(source_dir, target_dir):
    for file in os.listdir(source_dir):
        mask = cv2.imread(os.path.join(source_dir, file), cv2.IMREAD_GRAYSCALE)
        mask = change_masks(mask)
        cv2.imwrite(os.path.join(target_dir, file), mask)


if __name__ == "__main__":
    mkdir('datasets/nyu_navigation')
    mkdir('datasets/sun_navigation')

    mkdir('datasets/nyu_navigation/train_mask')
    mkdir('datasets/nyu_navigation/test_mask')
    mkdir('datasets/sun_navigation/train_mask')
    mkdir('datasets/sun_navigation/test_mask')

    transfer_mask_dir('datasets/nyu_v2/train_mask',
                      'datasets/nyu_navigation/train_mask')
    transfer_mask_dir('datasets/nyu_v2/test_mask',
                      'datasets/nyu_navigation/test_mask')

    # transfer_mask_dir('datasets/sun_rgbd/train_labels',
    #                   'datasets/sun_navigation/train_mask')
    # transfer_mask_dir('datasets/sun_rgbd/test_labels',
    #                   'datasets/sun_navigation/test_mask')
