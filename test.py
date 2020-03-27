import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2

l1 = cv2.imread(
    'datasets/sun_rgbd/test_labels/1.png', cv2.IMREAD_GRAYSCALE)
l2 = cv2.imread(
    'datasets/sun_rgbd/test_labels/2.png', cv2.IMREAD_GRAYSCALE)
l3 = cv2.imread(
    'datasets/sun_rgbd/test_labels/3.png', cv2.IMREAD_GRAYSCALE)
l4 = cv2.imread(
    'datasets/sun_rgbd/test_labels/4.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('lol', l4)
cv2.waitKey()
# l4 = K.variable(l4)
l4 = K.one_hot(l4, 38)
l1 = K.one_hot(l1, 38)
batch = K.stack((l1, l4))
print(batch.shape)

batch = K.argmax(batch)
print(batch.shape)
cv2.imshow('pute', batch.numpy().astype(np.uint8)[1])
cv2.waitKey()