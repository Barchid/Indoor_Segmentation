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
    def __init__(self, config):
        super(FpnNet, self).__init__(config)

    def build_model(self):
        # backbone encoder
        backbone, skips = self.get_backbone()

        # input layer
        input_layer = Input(shape=(self.config.model.height,
                                   self.config.model.width, input_channels), name="input_layer")
        print(input_layer.shape)

        network = keras.Model(
            inputs=input_layer, outputs=None, name="model_architecture")

        network.summary()
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

    def get_backbone(self):
        """Chooses the backbone model to use in the bottom-up pathway.
        Returns the backbone model and the outputs of the layers to use in skip connections.
        """
        # default network choice is mobilenet
        network = self.config.backbone if type(
            self.config.model.backbone) == str else 'mobilenet_v2'

        # constructor of backbone network
        backbone_fn = self.get_backbone_constructor(network)

        # shape of the input tensor
        input_shape = (
            self.config.model.height,
            self.config.model.width,
            self.config.model.channels
        )

        # pretrain on imagenet or not
        weights = 'imagenet' if self.config.model.imagenet_pretrain == 'imagenet' else None

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
            for layer_name in skip_connections[network]
        ]

        return backbone, skips

    def get_backbone_constructor(self, network):
        """Retrieves the constructor of the backbone chosen in config
        """
        if network == 'mobilenet_v2':
            backbone_fn = keras.applications.mobilenet_v2.MobileNetV2

        elif network == 'resnet18':
            raise NotImplementedError('ResNet18 needs implementation.')
        elif network == 'resnet50':
            backbone_fn = keras.applications.resnet.ResNet50

        elif network == 'resnet101':
            backbone_fn = keras.applications.resnet.ResNet101

        return backbone_fn
