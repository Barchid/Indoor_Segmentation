from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from layers.utils_layers import resize_img
from losses.focal_loss import CategoricalFocalLoss, BinaryFocalLoss
from losses.lovasz_softmax import MultiClassLovaszSoftmaxLoss
from tensorflow.keras import backend as K
from models.backbones.df import DF1, DF2
from layers.bi_fpn import BiFpnLayer, FastNormalizedFusion

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
    'EF0': (
        'block6a_expand_activation  ',  # stride 16
        'block4a_expand_activation',  # stride 8
        'block3a_expand_activation'  # stride 4
    ),
    'EF1': (
        'block6a_expand_activation  ',  # stride 16
        'block4a_expand_activation',  # stride 8
        'block3a_expand_activation'  # stride 4
    ),
    'EF2': (
        'block6a_expand_activation  ',  # stride 16
        'block4a_expand_activation',  # stride 8
        'block3a_expand_activation'  # stride 4
    ),
    'EF3': (
        'block6a_expand_activation  ',  # stride 16
        'block4a_expand_activation',  # stride 8
        'block3a_expand_activation'  # stride 4
    ),
    'EF4': (
        'block6a_expand_activation  ',  # stride 16
        'block4a_expand_activation',  # stride 8
        'block3a_expand_activation'  # stride 4
    )
}


class MudaNet(BaseModel):
    def __init__(self, config, datagen):
        super(MudaNet, self).__init__(config, datagen)

    def build_model(self):
        # backbone encoder
        backbone, skips = self.get_backbone()

        # respolution 1/32 (with PPM block module)
        P5 = ppm_block(backbone.output, (1, 2, 3, 6), 128, 256)
        P4 = skips[0]  # resolution 1/16
        P3 = skips[1]  # resolution 1/8
        P2 = skips[2]  # resolution 1/4

        outputs = []
        mini_decoders = []
        losses = {}
        loss_weights = {}
        metrics = {}

        # create a mini-decoder for each class of the segmentation task
        for cl in self.config.classes:
            name = cl.name
            P2_filters = cl.P2
            P3_filters = cl.P3
            P4_filters = cl.P4
            segmentation, mini_decoder = self.mini_decoder(
                P2, P3, P4, P5, name, P2_filters, P3_filters, P4_filters)

            mini_decoders.append(mini_decoder)
            outputs.append(segmentation)

            # loss & its weight for the segmentation mask
            losses[fix_output_name(segmentation.name)
                   ] = binary_seg_loss(cl.loss)
            loss_weights[fix_output_name(segmentation.name)] = cl.loss_weight

            # metric for the binary segmentation mask
            metrics[fix_output_name(segmentation.name)] = ['accuracy']

        # merging all mini-decoder's outputs
        main_out = self.merge_mini_decoders(mini_decoders)
        outputs.append(main_out)

        # register loss & its weight
        losses[fix_output_name(main_out.name)] = self.segmentation_loss()
        loss_weights[fix_output_name(
            main_out.name)] = self.config.model.loss_weights.main_segmentation

        # register metric for main out seg mask
        metrics[fix_output_name(main_out.name)] = self._generate_metrics()

        # create model
        network = keras.Model(
            inputs=backbone.input,
            outputs=outputs,
            name="MUDANET"
        )

        network.summary()

        # get the optimizer
        optimizer = self.build_optimizer()

        network.compile(
            loss=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics
        )

        return network

    def merge_mini_decoders(self, mini_decoders):
        """
        Merge all mini-decoders' outputs
        """
        merge = concatenate(mini_decoders, name="merge_block_concatenate")
        merge = conv2d(merge, self.config.model.classes, 1, 1,
                       kernel_size=3, name="merge_block_conv3x3")
        merge = conv2d(merge, self.config.model.classes, 1, 1,
                       kernel_size=3, name="merge_block_conv3x3_final")
        merge = Activation('softmax', name="Main_Softmax")(merge)
        merge = resize_img(merge, self.config.model.height,
                           self.config.model.width)
        return merge

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

    def mini_decoder(self, P2, P3, P4, P5, class_name, P2_filters, P3_filters, P4_filters):
        P5 = conv2d(P5, P4_filters, 1, 1, kernel_size=1,
                    name=class_name + "_decoder_control_backbone_")
        decoder = fusion_node(
            P5, P4, P4_filters, name="decoder_" + class_name + "_P5_P4_fusion_")
        decoder = fusion_node(
            decoder, P3, P3_filters, name="decoder_" + class_name + "_P4_P3_fusion_")
        decoder = fusion_node(
            decoder, P2, P2_filters, name="decoder_" + class_name + "_P3_P2_fusion_")

        # segmentation head
        segmentation = Conv2D(1, (1, 1), strides=(
            1, 1), padding="same", name=class_name + "_segout_conv")(decoder)
        segmentation = Activation(lambda a: tf.nn.sigmoid(
            a), name=class_name + "_sigmoid")(decoder)
        segmentation = resize_img(
            segmentation, self.config.model.height, self.config.model.width)

        return segmentation, decoder

    def segmentation_head(self, P2, P3, P4, P5, class_name=None, name="seg_head_"):
        features = conv2d(features, self.config.model.classes,
                          1, 1, kernel_size=1, use_bn=False, name=name + "conv1x1")

        upsampled = resize_img(
            features, self.config.model.height, self.config.model.width)

        segmentation_mask = Activation(
            'softmax', name=name + "out")(upsampled)
        return segmentation_mask

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
        x = conv2d(x, inter_channels, 1, 1, kernel_size=1)
        x = Lambda(lambda x: tf.image.resize(x, (H, W)))(x)
        concat.append(x)
    x = concatenate(concat)
    x = conv2d(x, out_channels, 1, 1, 3)
    return x


def conv2d(input, filters, stride, n, kernel_size=3, use_bn=True, name=None):
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

        if use_bn:
            x = BatchNormalization(name=bn_name)(x)

        x = Activation(lambda a: tf.nn.relu(a), name=relu_name)(x)
    return x


def fix_output_name(name: str):
    """Removes the "Identity:0" of a tensor's name if it exists"""
    return name.replace("/Identity:0", "", 1)


def fusion_node(low_res, high_res, filters=64, name="fusion_"):
    # channel controller
    high_res = conv2d(high_res, filters, 1, 1, kernel_size=1,
                      name=name + "channel_controller")
    low_res = UpSampling2D(name=name + "UP2D")(low_res)
    fusion = FastNormalizedFusion(
        name=name + "FastNormAdd")([low_res, high_res])
    fusion = conv2d(fusion, filters, 1, 1, kernel_size=3, name=name + "conv")
    return fusion


def binary_seg_loss(loss):
    """
    Chooses the binary segmentation loss to use depending on the loss name in parameter
    :param loss: the type of loss to use
    """
    if loss == 'focal':
        return BinaryFocalLoss()
    else:
        return tf.keras.losses.BinaryCrossentropy()
