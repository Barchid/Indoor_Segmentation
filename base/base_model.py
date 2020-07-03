from metrics.iou import build_iou_for, mean_iou
import tensorflow as tf
from tensorflow import keras
from metrics.softmax_miou import SoftmaxMeanIoU, SoftmaxSingleMeanIoU
import os
import tensorflow_addons as tfa
from losses.focal_loss import CategoricalFocalLoss
from losses.lovasz_softmax import MultiClassLovaszSoftmaxLoss


class BaseModel(object):
    def __init__(self, config, datagen):
        self.datagen = datagen
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
        # retrieve params from config
        momentum = self.config.model.optimizer.momentum
        initial_learning_rate = self.config.model.lr.initial
        power = self.config.model.lr.power
        cycle = self.config.model.lr.cycle
        policy = self.config.model.lr.policy

        # compute the total number of iterations through the epochs
        total_iterations = self.config.trainer.num_epochs * self.datagen.__len__()
        print('total iter', total_iterations)

        lr_policy = initial_learning_rate
        # poly learning rate policy
        if policy == "polynomial":
            lr_policy = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=total_iterations,
                power=power,
                cycle=cycle
            )
        elif policy == "cyclic":
            lr_policy = tfa.optimizers.CyclicalLearningRate(
                initial_learning_rate=initial_learning_rate,
                maximal_learning_rate=self.config.model.lr.maximal_learning_rate,
                step_size=self.config.model.lr.step_size,
                scale_fn=lambda x: 1.,
                scale_mode="cycle",
                name="CyclicalLearningRate"
            )
        elif policy == "triangular_cyclic":
            lr_policy = tfa.optimizers.TriangularCyclicalLearningRate(
                initial_learning_rate=initial_learning_rate,
                maximal_learning_rate=self.config.model.lr.maximal_learning_rate,
                step_size=self.config.model.lr.step_size,
                scale_mode='cycle',
                name='TriangularCyclicalLearningRate'
            )
        elif policy == "triangular2_cyclic":
            lr_policy = tfa.optimizers.Triangular2CyclicalLearningRate(
                initial_learning_rate=initial_learning_rate,
                maximal_learning_rate=self.config.model.lr.maximal_learning_rate,
                step_size=self.config.model.lr.step_size,
                scale_mode='cycle',
                name='Triangular2CyclicalLearningRate'
            )
        elif policy == "exp_cyclic":
            lr_policy = tfa.optimizers.ExponentialCyclicalLearningRate(
                initial_learning_rate=initial_learning_rate,
                maximal_learning_rate=self.config.model.lr.maximal_learning_rate,
                step_size=self.config.model.lr.step_size,
                scale_mode='cycle',
                gamma=self.config.model.lr.gamma,
                name='ExponentialCyclicalLearningRate'
            )

        # SGD optimizer
        sgd = keras.optimizers.SGD(
            learning_rate=lr_policy,
            momentum=momentum
        )
        return sgd

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

    def segmentation_loss(self):
        # define segmentation loss
        if self.config.model.seg_loss == "focal_loss":
            seg_loss = CategoricalFocalLoss(
                gamma=self.config.model.gamma,
                alpha=self.config.model.alpha
            )
        elif self.config.model.seg_loss == "lovasz":
            seg_loss = MultiClassLovaszSoftmaxLoss()
        else:
            seg_loss = 'categorical_crossentropy'

        return seg_loss

    def depth_loss(self):
        # define depth loss
        if self.config.model.depth_loss == "Huber":
            depth_loss = tf.keras.losses.Huber()
        else:
            depth_loss = tf.keras.losses.MeanSquaredError()

        return depth_loss
