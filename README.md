# Semantic segmentation for indoor RGB-D images

**Keywords :** Semantic segmentation, Deep Learning, Convolutional Neural Network, RGB-D images, SUN RGB-D, NYU-V2, SceneNet RGB-D, Computer Vision, Neural Networks, Tensorflow, Keras.

## Introduction
This repository contains the codes and models to address the problem of semantic segmentation for indoor scenes in real-time. The project is written in Keras and Tensorflow 2.1.

## Dependencies
This code was used with **Python 3.7.6** Install the dependencies with :

``pip install -r requirements.txt``


## Install the datasets
The project uses three datasets well known in the indoor semantic segmentation problem : **NYU-v2**, **SUN RGB-D** and **SceneNet RGB-D**.

Execute the bash script ``install.sh`` located in the project root in order to install and format the datasets.

## Usage
You can use one of the available models of this repository by executing a python script with a JSON config file that describes all the parameters used by the model (e.g. the learning rate, the dataset's directory, ...). For exemple : 

```
python light-scnn.py -c configs/segmentation-example.json
```

## Project architecture
The project contains a customizable structure you can use to create your personal models :

```
├── main.py             - here's an example of main that is responsible for the whole pipeline. It instantiates and executes all the other components
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_model.py   - abstract class to inherit in order to define the new model architecture.
│   └── base_train.py   - this file contains the abstract class of the trainer to inherit to define a new trainer architecture.
│
│
├── models               - this folder contains the models of your project.
│
├── trainers             - this folder contains the trainers of your project.
│   ├── simple_mnist_trainer.py             - Example for MNIST classification training
│   └── segmentation_trainer.py             - General trainer to use for segmentation task.
│
|
├── configs                                 - this folder contains the experiment and model configs of your project.
│   ├── simple_mnist_config.json            - example for simple MNIST classification task.
│   └── segmentation_example.json           - another example for segmentation task.
│
├── datasets                                - this folder might contain the datasets of your project.
│   ├── sun_rgbd                            - contains the SUN RGB-D dataset.
│   ├── nyu_v2                              - contains the NYU-V2 dataset.
│   ├── scenenet_rgbd                       - contains the ScenetNet RGB-D dataset.
│   └── stanford_indoor                     - contains the Stanford 2D-3D-S RGB-D dataset.
│
├── evaluater                               - this folder contains the evaluation for a segmentation task using the selected test set.
│
├── data_generators                         - this folder contains all the process of data generation for a segmentation task training.
│
├── layers                                  - this folder contains some Keras layers that can be used in the models architecture.
│
├── metrics                                 - this folder contains several implementation of metrics to use in our Keras architecture.
│
└── utils               - this folder contains any utils you need.
     ├── config.py      - util functions for parsing the config files.
     ├── dirs.py        - util functions for creating directories.
     └── utils.py       - util functions for parsing arguments.
```

## Customization 
You can adapt this project structure to use create a custom model or use another dataset.

### Create a new model
In order to create a model, you have to :
- Create a model that inherits `BaseModel` and constructs the architecture. *Note : examples available in `models/` directory*.
- Create a JSON config file with all the required options. Option fields can be added and will be available during the execution of the models. *Note : examples available in `configs/` directory.*
- Create a python main script that instantiates at least the required components (a `trainer`, a `model` & the `SegmentationDataGenerator`). Don't hesitate to read the examples available.

### Add another dataset
The `SegmentationDataGenerator` class can perform data generation and data augmentation for a segmentation training but requires the new dataset to have the right structure. 

Assuming there is three kinds of image inputs for the model : the input images (RGB), the associated depth images and the associated mask images. The dataset's training and testing sets have to be split up into 3 folders : one for the training RGB images, one for the depth images and one for the mask images. The below example shows the kind of nomenclature that has to be used :

```
├── RGB_train_set                               - this folder contains the RGB images used in the training set
│   ├── 1.jpg                                   - example image of the training set.
...
...
├── mask_train_set                              - this folder contains the mask labels images used in the training set.
│   ├── 1.png                                   - Mask image related to the RGB image '1.jpg'. WARNING : the filename must be the same.
...
...
├── depth_train_set                             - this folder contains the depth images used in the training set.
│   ├── 1.png                                   - depth images related to the RGB image '1.jpg'. WARNING : the filename must be the same.
...
...
├── RGB_test_set                                - this folder contains the RGB images used in the testing set
│   ├── 5.jpg                                   - example image of the testing set.
...
...
├── mask_test_set                               - this folder contains the mask labels images used in the testing set.
│   ├── 5.png                                   - Mask image related to the testing RGB image '5.jpg'. WARNING : the filename must be the same.
...
...
├── depth_test_set                              - this folder contains the depth images used in the testing set.
│   ├── 5.png                                   - depth images related to the testing RGB image '5.jpg'. WARNING : the filename must be the same.
...
...

```

## Features available
- Data generation for segmentation dataset
- Data augmention with the following random operations :
    - Temperature change
    - Brightness modification
    - Horizontal flipping
    - Cropping
- Tensorboard integration (Keras callback)
```shell
tensorboard --logdir=experiments/simple_mnist/logs
```
- Training + evaluation
- Load/Save weights file (Keras callback)