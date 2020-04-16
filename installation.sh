#!/bin/bash

####################################################################
####################################################################
# @author : BARCHID Sami - IMT Lille-Douai
#       This script aims to install the environment to make it work.
#       It downloads and parses the dataset.
####################################################################
####################################################################

echo "########################################################################\nCreate datasets\n########################################################################"
mkdir datasets

echo "########################################################################\nCreate SUN-RGBD dataset\n########################################################################"
# mkdir datasets/sun_rgbd
# cd datasets/sun_rgbd

# mkdir SUNRGBD-train_images
# wget http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-train_images.tgz
# tar -xzf SUNRGBD-train_images.tgz -C SUNRGBD-train_images

# mkdir SUNRGBD-test_images
# wget http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz
# tar -xzf SUNRGBD-test_images.tgz -C SUNRGBD-test_images

# mkdir sunrgbd_train_test_labels
# wget https://github.com/ankurhanda/sunrgbd-meta-data/raw/master/sunrgbd_train_test_labels.tar.gz
# tar -xzf sunrgbd_train_test_labels.tar.gz -C sunrgbd_train_test_labels

# wget https://www.doc.ic.ac.uk/~ahanda/sunrgb_train_depth.tgz
# tar -xzf sunrgb_train_depth.tgz

# wget https://www.doc.ic.ac.uk/~ahanda/sunrgb_test_depth.tgz
# tar -xzf sunrgb_test_depth.tgz

# cd ../..
# python sun_rgbd_dataset_parser.py

echo "\n########################################################################\nCreate NYU-V2 dataset\n########################################################################\n"
tar -xf nyu_labels_40.tar.xz
tar -xf nyu_depths.tar.xz
mkdir datasets/nyu_v2

mv labels_40 datasets/nyu_v2/
mv train_depth datasets/nyu_v2/
mv test_depth datasets/nyu_v2/

cd datasets/nyu_v2
wget http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
mkdir train_rgb
mv nyu_train_rgb.tgz train_rgb
cd train_rgb
tar -xzf nyu_train_rgb.tgz
cd ..

wget http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
mkdir test_rgb
mv nyu_test_rgb.tgz test_rgb
cd test_rgb
tar -xzf nyu_test_rgb.tgz
cd ..

cd ../..
python nyu_v2_dataset_parser.py