import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2
from evaluater.functional import *

l1 = cv2.imread(
    'datasets/sun_rgbd/test_labels/1.png', cv2.IMREAD_GRAYSCALE)
l2 = cv2.imread(
    'datasets/sun_rgbd/test_labels/2.png', cv2.IMREAD_GRAYSCALE)
l3 = cv2.imread(
    'datasets/sun_rgbd/test_labels/3.png', cv2.IMREAD_GRAYSCALE)
l4 = cv2.imread(
    'datasets/sun_rgbd/test_labels/4.png', cv2.IMREAD_GRAYSCALE)

# l4 = K.variable(l4)
l1 = K.one_hot(l1, 38)
l2 = K.one_hot(l2, 38)
l3 = K.one_hot(l3, 38)
l4 = K.one_hot(l4, 38)
y_true = K.stack((l1, l2))
y_pred = K.stack((l3, l4))

lol = iou_score(y_true, y_pred)
print(lol)

label = 0
print(y_true.shape, y_pred.shape)

y_true = K.cast(K.equal(K.argmax(y_true), label), tf.float32)
y_pred = K.cast(K.equal(K.argmax(y_pred), label), tf.float32)
# print(y_true.shape, y_pred.shape, y_true)

intersection = K.sum(y_true * y_pred)
union = K.sum(y_true + y_pred) - intersection
iou = (intersection + K.epsilon()) / (union + K.epsilon())
print(union)
print(iou)
print(intersection.shape, intersection)
