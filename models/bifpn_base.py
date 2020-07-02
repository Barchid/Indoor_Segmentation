from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from layers.utils_layers import resize_img
from losses.focal_loss import CategoricalFocalLoss
from losses.lovasz_softmax import MultiClassLovaszSoftmaxLoss
from tensorflow.keras import backend as K
from models.backbones.df import DF1, DF2
from layers.bi_fpn import BiFpnLayer

# name of the layers used for the skip connections in the top-down pathway
skip_connections = {
    'mobilenet_v2': (
        'block_13_expand_relu',  # stride 16
        'block_6_expand_relu',  # stride 8
        'block_3_expand_relu'   # stride 4
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
    ),
    'DF1': (
        'tf_op_layer_Relu_13',  # stride 16
        'tf_op_layer_Relu_7',  # stride 8
        'tf_op_layer_Relu_1'  # stride 4
    ),
    'DF2': (
        'tf_op_layer_Relu_29',  # stride 16
        'tf_op_layer_Relu_7',  # stride 8
        'tf_op_layer_Relu_1'  # stride 4
    ),
}


class BiFpnBase(BaseModel):
    def __init__(self, config, datagen):
        super(BiFpnBase, self).__init__(config, datagen)

    def build_model(self):
        # backbone encoder
        backbone, skips = self.get_backbone()

        # decoder construction
        prediction = self.create_decoder(backbone, skips)

        network = keras.Model(
            inputs=backbone.input, outputs=prediction, name="BI_FPN_NET")

        network.summary()

        # get the optimizer
        optimizer = self.build_optimizer()

        metrics = self.build_metrics_NYU()

        if self.config.model.loss == "focal_loss":
            loss = CategoricalFocalLoss(
                gamma=self.config.model.gamma,
                alpha=self.config.model.alpha
            )
        elif self.config.model.loss == "lovasz":
            loss = MultiClassLovaszSoftmaxLoss()
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
        name = self.config.model.backbone if type(
            self.config.model.backbone) == str else 'mobilenet_v2'

        # constructor of backbone network
        backbone_fn = self.get_backbone_constructor(name)

        # shape of the input tensor
        input_tensor = Input(shape=(
            self.config.model.height,
            self.config.model.width,
            self.config.model.channels
        ), name="input_layer_backbone")

        # pretrain on imagenet or not
        weights = 'imagenet' if self.config.model.backbone_pretrain == 'imagenet' else None

        # instantiate the backbone network
        backbone = backbone_fn(
            input_tensor=input_tensor,
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
        # default is MobileNetV2
        backbone_fn = keras.applications.mobilenet_v2.MobileNetV2

        if name == 'resnet18':
            raise NotImplementedError('ResNet18 needs implementation.')
        elif name == 'resnet50':
            backbone_fn = keras.applications.resnet.ResNet50

        elif name == 'resnet101':
            backbone_fn = keras.applications.resnet.ResNet101
        elif name == 'DF1':
            backbone_fn = DF1
        elif name == 'DF2':
            backbone_fn = DF2

        return backbone_fn

    def create_decoder(self, backbone, skips):
        stage5 = backbone.output  # resolution 1/32
        stage4 = skips[0]  # resolution 1/16
        stage3 = skips[1]  # resolution 1/8
        stage2 = skips[2]  # resolution 1/4

        # Pyramid pooling module for stage 5 tensor
        stage5 = ppm_block(stage5, (1, 2, 3, 6), 128, 512)

        # 3x BI_FPN blocks
        P2, P3, P4, P5 = BiFpnLayer(
            stage2, stage3, stage4, stage5, filters=256, conv_input=True, name="BiFpn_0_")
        P2, P3, P4, P5 = BiFpnLayer(
            P2, P3, P4, P5, filters=256, conv_input=True, name="BiFpn_1_")
        P2 = BiFpnLayer(
            P2, P3, P4, P5, filters=256, conv_input=True, has_bottomup=False, name="BiFpn_2_")

        prediction = self.segmentation_head(P2)
        return prediction

    def segmentation_head(self, features):
        features = conv2d(features, 64, 1, 1, kernel_size=3,
                          name="segmentation_head_3x3")
        features = conv2d(features, self.config.model.classes,
                          1, 1, kernel_size=1, name="segmentation_head_1x1")

        upsampled = resize_img(
            features, self.config.model.height, self.config.model.width)

        segmentation_mask = Activation(
            'softmax', name='seg_out')(upsampled)
        return segmentation_mask

# Layer functions


def merge_block(stage4, stage3, stage2, out_channels):
    # get fusion2 channels number
    inter_channels = K.int_shape(stage2)[-1]

    # fusion3 upsampling
    stage3 = conv2d(stage3, inter_channels, 1, 1, kernel_size=1)
    stage3 = UpSampling2D(size=(2, 2))(stage3)

    # fusion4 upsampling
    stage4 = conv2d(stage4, inter_channels, 1, 1, kernel_size=1)
    stage4 = UpSampling2D(size=(4, 4))(stage4)

    # addition
    merge = Add()([
        stage4,
        stage3,
        stage2
    ])

    # last conv layer
    merge = conv2d(merge, out_channels, 1, 1, kernel_size=3, use_relu=True)
    return merge


def ppm_block(input, bin_sizes, inter_channels, out_channels):
    """
    Pyramid pooling module with bins (1, 2, 3 and 6)
    :param input: the feature map input
    :param bin_sizes: list of sizes of pooling
    :param inter_channels: number of channels for each pooled bin
    :param out_channels: total number of channels of the output tensor
    """
    concat = [input]
    H = K.int_shape(input)[1]
    W = K.int_shape(input)[2]

    for bin_size in bin_sizes:
        x = AveragePooling2D(
            pool_size=(H//bin_size, W//bin_size),
            strides=(H//bin_size, W//bin_size)
        )(input)
        x = conv2d(x, inter_channels, 1, 1, kernel_size=1)
        x = Lambda(lambda x: tf.image.resize(x, (H, W)))(x)
        concat.append(x)
    x = concatenate(concat)
    x = conv2d(x, out_channels, 1, 1, 3)
    return x


def conv2d(input, filters, stride, n, kernel_size=3, name=None):
    x = input
    for i in range(n):
        # define names for layers
        if name is not None:
            conv_name = name + "_conv_" + str(i)
            bn_name = name + "_bn_" + str(i)
            relu_name = name + "_relu_" + str(i)
        else:
            conv_name = bn_name = relu_name = None

        x = Conv2D(filters, (kernel_size, kernel_size), strides=(
            stride, stride), padding="same", name=conv_name)(x)

        x = BatchNormalization(name=bn_name)(x)

        x = Activation(lambda a: tf.nn.relu(a), name=relu_name)(x)
    return x
