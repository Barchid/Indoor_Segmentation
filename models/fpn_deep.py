from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from layers.utils_layers import resize_img
from losses.focal_loss import CategoricalFocalLoss
from losses.lovasz_softmax import MultiClassLovaszSoftmaxLoss
from tensorflow.keras import backend as K
from models.backbones.df import DF1, DF2

# name of the layers used for the skip connections in the top-down pathway
skip_connections = {
    'mobilenet_v2': (
        'block_13_expand_relu',  # stride 16
        'block_6_expand_relu',  # stride 8
        'block_3_expand_relu',   # stride 4
        'block_1_expand_relu'   # stride 2
    ),
    'resnet18': (

    ),
    'resnet50': (
        'conv4_block6_out',  # stride 16
        'conv3_block4_out',  # stride 8
        'conv2_block3_out',  # stride 4
        'conv1_relu'  # stride 2
    ),
    'resnet101': (
        'conv4_block23_out',  # stride 16
        'conv3_block4_out',  # stride 8
        'conv2_block3_out'  # stride 4
    ),
    'DF1': (
        'tf_op_layer_Relu_13',  # stride 16
        'tf_op_layer_Relu_7',  # stride 8
        'tf_op_layer_Relu_1',  # stride 4
        'tf_op_layer_Relu'  # stride 2
    ),
    'DF2': (
        'tf_op_layer_Relu_29',  # stride 16
        'tf_op_layer_Relu_7',  # stride 8
        'tf_op_layer_Relu_1',  # stride 4
        'tf_op_layer_Relu'  # stride 2
    ),
}


class FpnDeep(BaseModel):
    def __init__(self, config, datagen):
        super(FpnDeep, self).__init__(config, datagen)

    def build_model(self):
        # backbone encoder
        backbone, skips = self.get_backbone()

        # decoder construction
        prediction = self.create_decoder(backbone, skips)

        network = keras.Model(
            inputs=backbone.input, outputs=prediction, name="FPN_NET")

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
        stage1 = skips[3]  # resolution 1/2

        # Pyramid pooling module for stage 5 tensor
        stage5 = ppm_block(stage5, (1, 2, 3, 6), 128, 512)

        # channel controllers
        skip4 = conv2d(stage4, 254, 1, 1, kernel_size=1, use_relu=True)
        skip3 = conv2d(stage3, 128, 1, 1, kernel_size=1, use_relu=True)
        skip2 = conv2d(stage2, 64, 1, 1, kernel_size=1, use_relu=True)
        skip1 = conv2d(stage1, self.config.model.classes,
                       1, 1, kernel_size=1, use_relu=True)

        # fusion nodes
        fusion = fusion_node(stage5, skip4)
        fusion = fusion_node(fusion, skip3)
        fusion = fusion_node(fusion, skip2)
        fusion = fusion_node(fusion, skip1)
        print(fusion.shape)

        # upsample to the right dimensions
        upsampled = resize_img(
            fusion, self.config.model.height, self.config.model.width)
        prediction = Activation('softmax', dtype='float32')(upsampled)
        return prediction

# Layer functions


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
        x = conv2d(x, inter_channels, 1, 1, kernel_size=1, use_relu=True)
        x = Lambda(lambda x: tf.image.resize(x, (H, W)))(x)
        concat.append(x)
    x = concatenate(concat)
    x = conv2d(x, out_channels, 1, 1, 3, use_relu=True)
    return x


def fusion_node(low_res, high_res):
    # channel depth of output is high_res feature map depth
    out_channels = K.int_shape(high_res)[-1]

    # upsample the deepest, low-resolution feature map
    low_res = conv2d(low_res, out_channels, 1, 1, kernel_size=1, use_relu=True)
    low_res = UpSampling2D(size=(2, 2))(low_res)

    # fusion of two tensors
    fusion = concatenate([low_res, high_res])
    fusion = conv2d(fusion, out_channels, 1, 1, kernel_size=3, use_relu=True)

    return fusion


def ds_conv2d(input, filters, stride, n, kernel_size=3):
    """Performs n times separable convolution + BN + relu operation
    :param input: input tensor
    :param filters: number of filters
    :param stride: stride
    :param n: number of times the operation is performed
    :param kernel_size: dimension size of a filter. Default = 3x3.
    """
    x = input
    for i in range(n):
        x = SeparableConv2D(filters, (kernel_size, kernel_size),
                            strides=(stride, stride), padding="same")(x)
        x = BatchNormalization()(x)
        x = keras.activations.relu(x)
    return x


def conv2d(input, filters, stride, n, kernel_size=3, use_relu=True):
    x = input
    for i in range(n):
        x = Conv2D(filters, (kernel_size, kernel_size),
                   strides=(stride, stride), padding="same")(x)
        x = BatchNormalization()(x)
        if use_relu:
            x = keras.activations.relu(x)
    return x
