from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from layers.utils_layers import resize_img
from losses.focal_loss import CategoricalFocalLoss

# name of the layers used for the skip connections in the top-down pathway
skip_connections = {
    'mobilenet_v2': (
        'block_13_expand_relu',  # stride 16
        'block_6_expand_relu',  # stride 8
        'block_3_expand_relu'  # stride 4
    ),
    'resnet18': (

    ),
    'resnet50': (
        'conv4_block6_out',  # stride 16
        'conv3_block4_out',  # stride 8
        'conv2_block3_out'  # stride 4
    ),
    'resnet101': (
        'conv4_block23_out',  # stride 16
        'conv3_block4_out',  # stride 8
        'conv2_block3_out'  # stride 4
    )
}


class FpnNet(BaseModel):
    def __init__(self, config, datagen):
        super(FpnNet, self).__init__(config)
        self.datagen = datagen

    def build_model(self):
        # backbone encoder
        backbone, skips = self.get_backbone()

        # decoder construction

        network = keras.Model(
            inputs=backbone.input, outputs=None, name="FPN_NET")

        network.summary()

        # get the optimizer
        optimizer = self.build_optimizer()

        metrics = self.build_metrics_NYU()

        if self.config.model.loss == "focal_loss":
            loss = CategoricalFocalLoss(
                gamma=self.config.model.gamma,
                alpha=self.config.model.alpha
            )
        else:
            loss = 'categorical_crossentropy'

        network.compile(loss=loss,
                        optimizer=optimizer, metrics=metrics)

        return network

    def build_optimizer(self):
        # retrieve params from config
        momentum = self.config.model.optimizer.momentum
        initial_learning_rate = self.config.model.lr.initial
        power = self.config.model.lr.power
        cycle = self.config.model.lr.cycle

        # compute the total number of iterations through the epochs
        total_iterations = self.config.trainer.num_epochs * self.datagen.__len__()

        # poly learning rate policy
        poly_lr_policy = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_iterations,
            power=power,
            cycle=cycle
        )

        # SGD optimizer
        sgd = keras.optimizers.SGD(
            lr=poly_lr_policy,
            momentum=momentum
        )
        return sgd

    def get_backbone(self):
        """Chooses the backbone model to use in the bottom-up pathway.
        Returns the backbone model and the outputs of the layers to use in skip connections.
        """
        # default network choice is mobilenet
        name = self.config.backbone if type(
            self.config.model.backbone) == str else 'mobilenet_v2'

        # constructor of backbone network
        backbone_fn = self.get_backbone_constructor(name)

        # shape of the input tensor
        input_shape = (
            self.config.model.height,
            self.config.model.width,
            self.config.model.channels
        )

        # pretrain on imagenet or not
        weights = 'imagenet' if self.config.model.backbone_pretrain == 'imagenet' else None

        # instantiate the backbone network
        backbone = backbone_fn(
            input_shape=input_shape,
            weights=weights,
            include_top=False,
            pooling=None
        )

        # retrieve skip layer's output
        skips = [
            backbone.get_layer(layer_name).output
            for layer_name in skip_connections[name]
        ]

        return backbone, skips

    def get_backbone_constructor(self, name):
        """Retrieves the constructor of the backbone chosen in config
        """
        if name == 'mobilenet_v2':
            backbone_fn = keras.applications.mobilenet_v2.MobileNetV2

        elif name == 'resnet18':
            raise NotImplementedError('ResNet18 needs implementation.')
        elif name == 'resnet50':
            backbone_fn = keras.applications.resnet.ResNet50

        elif name == 'resnet101':
            backbone_fn = keras.applications.resnet.ResNet101

        return backbone_fn
