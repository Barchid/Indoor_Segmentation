"""Evaluates the segmentation task. Takes the test dataset paths in parameter and create the class-specific mIoUs, the general mIoU and the accuracy.
"""
import argparse
from utils.utils import get_args
from utils.config import process_config
from evaluater.evaluation import evaluation


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args(test_args=True)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Force some parameters from configuration
    config.generator.use_data_augmentation = False
    config.trainer.batch_size = 1

    # Launch evaluation
    evaluation(config, visualization=args.visualize)


if __name__ == "__main__":
    main()
