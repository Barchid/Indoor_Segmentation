#!/bin/bash

####################################################################
####################################################################
# @author : BARCHID Sami - IMT Lille-Douai
#       This script aims to install the environment to make it work.
#       It downloads and parses the dataset.
####################################################################
####################################################################

echo "########################################################################\nCreate SUN-RGBD dataset\n########################################################################"
mkdir datasets
cd datasets
mkdir sun_rgbd
cd sun_rgbd

mkdir SUNRGBD-train_images
wget http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-train_images.tgz
tar -xzf SUNRGBD-train_images.tgz -C SUNRGBD-train_images

mkdir SUNRGBD-test_images
wget http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz
tar -xzf SUNRGBD-test_images.tgz -C SUNRGBD-test_images

mkdir sunrgbd_train_test_labels
wget https://github.com/ankurhanda/sunrgbd-meta-data/raw/master/sunrgbd_train_test_labels.tar.gz
tar -xzf sunrgbd_train_test_labels.tar.gz -C sunrgbd_train_test_labels

wget https://www.doc.ic.ac.uk/~ahanda/sunrgb_train_depth.tgz
tar -xzf sunrgb_train_depth.tgz

wget https://www.doc.ic.ac.uk/~ahanda/sunrgb_test_depth.tgz
tar -xzf sunrgb_test_depth.tgz

python sun_rgbd_dataset_parser.py