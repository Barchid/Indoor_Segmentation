from metrics.iou import build_iou_for, mean_iou
import tensorflow as tf
from tensorflow import keras
from metrics.softmax_miou import SoftmaxMeanIoU, SoftmaxSingleMeanIoU
import os


class BaseModel(object):
    def __init__(self, config):
        self.config = config

        # COLAB TPU USAGE if available
        strategy = None
        if 'COLAB_TPU_ADDR' not in os.environ:
            print(
                'Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
        else:
            tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            print('TPU address is', tpu_address)
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu=tpu_address)
            tf.config.experimental_connect_to_cluster(resolver)
            # This is the TPU initialization code that has to be at the beginning.
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)

        if strategy is None:
            self.model = self.build_model()
        else:
            with strategy.scope():
                self.model = self.build_model()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError

    def build_optimizer(self):
        """Builds the optimizer according to the config file
        """
        if hasattr(self.config.model, "optimizer") and self.config.model.optimizer == 'SGD':
            return keras.optimizers.SGD(momentum=self.config.model.momentum, lr=self.config.model.learning_rate)
        elif hasattr(self.config.model, "optimizer") and self.config.model.optimizer == 'Adam':
            return keras.optimizers.Adam(
                learning_rate=self.config.model.learning_rate,
                beta_1=self.config.model.beta_1,
                beta_2=self.config.model.beta_2
            )
        else:
            raise Exception('No model.optimizer found in JSON config file')

    def build_metrics_SUN(self):
        """Generates the list of metrics to evaluate with SUN RGB-D.
        """
        metrics = self._generate_metrics()
        # ious = build_iou_for(
        #     label=[*range(38)],
        #     name=[
        #         "background",  # 0
        #         "wall",  # 1
        #         "floor",  # 2
        #         "cabinet",  # 3
        #         "bed",  # 4
        #         "chair",  # 5
        #         "sofa",  # 6
        #         "table",  # 7
        #         "door",  # 8
        #         "window",  # 9
        #         "bookshelf",  # 10
        #         "picture",  # 11
        #         "counter",  # 12
        #         "blinds",  # 13
        #         "desk",  # 14
        #         "shelves",  # 15
        #         "curtain",  # 16
        #         "dresser",  # 17
        #         "pillow",  # 18
        #         "mirror",  # 19
        #         "floor_mat",  # 20
        #         "clothes",  # 21
        #         "ceiling",  # 22
        #         "books",  # 23
        #         "fridge",  # 24
        #         "tv",  # 25
        #         "paper",  # 26
        #         "towel",  # 27
        #         "shower_curtain",  # 28
        #         "box",  # 29
        #         "whiteboard",  # 30
        #         "person",  # 31
        #         "night_stand",  # 32
        #         "toilet",  # 33
        #         "sink",  # 34
        #         "lamp",  # 35
        #         "bathtub",  # 36
        #         "bag"  # 37
        #     ]
        # )
        # metrics.extend(ious)
        return metrics

    def build_metrics_NYU(self):
        """Generates the metrics for the NYU-v2 dataset
        """
        metrics = self._generate_metrics()
        # ious = build_iou_for(
        #     label=[*range(41)],
        #     name=[
        #         "void",  # 0
        #         "wall",  # 1
        #         "floor",  # 2
        #         "cabinet",  # 3
        #         "bed",  # 4
        #         "chair",  # 5
        #         "sofa",  # 6
        #         "table",  # 7
        #         "door",  # 8
        #         "window",  # 9
        #         "bookshelf",  # 10
        #         "picture",  # 11
        #         "counter",  # 12
        #         "blinds",  # 13
        #         "desk",  # 14
        #         "shelves",  # 15
        #         "curtain",  # 16
        #         "dresser",  # 17
        #         "pillow",  # 18
        #         "mirror",  # 19
        #         "floor_mat",  # 20
        #         "clothes",  # 21
        #         "ceiling",  # 22
        #         "books",  # 23
        #         "fridge",  # 24
        #         "tv",  # 25
        #         "paper",  # 26
        #         "towel",  # 27
        #         "shower_curtain",  # 28
        #         "box",  # 29
        #         "whiteboard",  # 30
        #         "person",  # 31
        #         "night_stand",  # 32
        #         "toilet",  # 33
        #         "sink",  # 34
        #         "lamp",  # 35
        #         "bathtub",  # 36
        #         "bag",  # 37
        #         "otherstructure",       # 38
        #         "otherfurniture",       # 39
        #         "otherprop",   # 40
        #     ]
        # )
        # metrics.extend(ious)
        return metrics

    def build_metrics_navigation(self):
        """Generates the metrics for the NYU-v2 dataset
        """
        metrics = self._generate_metrics()
        ious = build_iou_for(
            label=[*range(5)],
            name=[
                "void",  # 0
                "wall",  # 1
                "floor",  # 2
                "person",  # 3
                "other"  # 4
            ]
        )
        metrics.extend(ious)
        return metrics

    def _generate_metrics(self):
        metrics = []
        # metrics.append(mean_iou)  # general mIoU
        metrics.append(SoftmaxMeanIoU(
            num_classes=self.config.model.classes, name='Mean_IoU'))
        metrics.append('accuracy')
        return metrics
