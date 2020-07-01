from utils import factory
from utils.config import process_config
import os
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

MIN_LR = 1e-10
MAX_LR = 1e-1
BATCH_SIZE = 16
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 48
