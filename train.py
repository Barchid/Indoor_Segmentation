from models.fpn_net import FpnNet
from trainers.segmentation_trainer import SegmentationTrainer
from data_generators.segmentation_data_generator import SegmentationDataGenerator
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import os
import pprint

tf.config.optimizer.set_jit(True)

# COLAB TPU USAGE
is_tpu_colab = False
if 'COLAB_TPU_ADDR' not in os.environ:
    print('Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
else:
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    print('TPU address is', tpu_address)
    is_tpu_colab = True

    with tf.Session(tpu_address) as session:
        devices = session.list_devices()

    print('TPU devices:')
    pprint.pprint(devices)


def main():
        # capture the config path from the run arguments
        # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # use mixed precision for training
    if config.exp.mixed_precision:
        print('Use mixed precision training')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir,
                 config.callbacks.checkpoint_dir])

    print('Create the training data generator.')
    train_data = SegmentationDataGenerator(config)

    validation_data = None
    if type(config.validation.img_dir) == str:
        print('Create the validation data generator.')
        validation_data = SegmentationDataGenerator(
            config, is_training_set=False)

    print('Create the model.')
    model = FpnNet(config, train_data)

    # use TPU from colab if available
    if is_tpu_colab:
        # This address identifies the TPU we'll use when configuring TensorFlow.
        TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        tf.logging.set_verbosity(tf.logging.INFO)
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

    print('Create the trainer')
    trainer = SegmentationTrainer(
        model, train_data, config, validation_generator=validation_data)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
