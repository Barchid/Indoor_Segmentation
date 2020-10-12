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

# echo "########################################################################\nCreate SUN-RGBD dataset\n########################################################################"
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
rm nyu_train_rgb.tgz
cd ..

wget http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
mkdir test_rgb
mv nyu_test_rgb.tgz test_rgb
cd test_rgb
tar -xzf nyu_test_rgb.tgz
rm nyu_test_rgb.tgz
cd ..

cd ../..
python nyu_v2_dataset_parser.py


echo "\n########################################################################\nCreate NYU-V2 dataset 13 classes\n########################################################################\n"

mkdir datasets/nyu_v2_13c
mkdir datasets/nyu_v2_13c/train_mask
mkdir datasets/nyu_v2_13c/test_mask
cp -r datasets/nyu_v2/train_rgb datasets/nyu_v2_13c
cp -r datasets/nyu_v2/train_depth datasets/nyu_v2_13c
cp -r datasets/nyu_v2/test_rgb datasets/nyu_v2_13c
cp -r datasets/nyu_v2/test_depth datasets/nyu_v2_13c


mv nyuv2_train_class13.tgz datasets/nyu_v2_13c/train_mask
mv nyuv2_test_class13.tgz datasets/nyu_v2_13c/test_mask
cd datasets/nyu_v2_13c/train_mask
tar -xzf nyuv2_train_class13.tgz
cd ../../..

cd datasets/nyu_v2_13c/test_mask
tar -xzf nyuv2_test_class13.tgz
cd ../../..
rm datasets/nyu_v2_13c/train_mask/nyuv2_train_class13.tgz
rm datasets/nyu_v2_13c/test_mask/nyuv2_test_class13.tgz

python nyu_v2_13c_dataset_parser.py


# echo "\n########################################################################\nCreate navigation datasets\n########################################################################\n"
# python navigation_datasets_parser.py