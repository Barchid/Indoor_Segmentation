from data_generators.joint_data_generator import JointDataGenerator
from data_generators.depth_data_generator import DepthDataGenerator
from data_generators.segmentation_data_generator import SegmentationDataGenerator
from utils import factory
from utils.config import process_config
from utils.utils import get_args
from utils.lr_finder import LearningRateFinder
import matplotlib.pyplot as plt
import os
import numpy as np
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# capture the config path from the run arguments
# then process the json configuration file
try:
    args = get_args()
    config = process_config(args.config)
except:
    print("missing or invalid arguments")
    exit(0)

MIN_LR = 1e-10
MAX_LR = 1.
NUM_EPOCHS = 48
MODEL_TASK = "RGB"

datagen = None
if MODEL_TASK == "RGB":
    datagen = SegmentationDataGenerator(config)
elif MODEL_TASK == "D":
    datagen = DepthDataGenerator(config)
else:
    datagen = JointDataGenerator(config)

network = factory.create(config.model.class_name)(config, datagen)

# initialize the learning rate finder and then train with learning
# rates ranging from 1e-10 to 1e+1
print("[INFO] finding learning rate...")
lrf = LearningRateFinder(network.model)
lrf.find(
    datagen,
    MIN_LR,
    MAX_LR,
    stepsPerEpoch=datagen.__len__(),
    batchSize=config.trainer.batch_size,
    epochs=NUM_EPOCHS
)

# plot the loss for the various learning rates and save the
# resulting plot to disk
lrf.plot_loss()
plt.savefig("lr_finder.png")

print("[INFO] learning rate finder complete")
print("[INFO] examine plot and adjust learning rates before training")
