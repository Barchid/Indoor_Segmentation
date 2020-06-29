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


class JointFpnDeep(BaseModel):
    def __init__(self, config, datagen):
        self.datagen = datagen
        super(JointFpnDeep, self).__init__(config)

    def build_model(self):
        # backbone encoder
        backbone, skips = self.get_backbone()

        # joint decoder (resolution 1/4)
        decoder = self.decoder(backbone, skips)

        # segmentation head
        segmentation_mask = self.segmentation_head(decoder)

        # depth estimation head
        depth_estimation = self.depth_head(decoder)

        # create model
        network = keras.Model(
            inputs=backbone.input,
            outputs=[
                segmentation_mask,
                depth_estimation
            ],
            name="JOINT_FPN"
        )

        network.summary()

        # get the optimizer
        optimizer = self.build_optimizer()

        # define segmentation loss
        if self.config.model.loss == "focal_loss":
            seg_loss = CategoricalFocalLoss(
                gamma=self.config.model.gamma,
                alpha=self.config.model.alpha
            )
        elif self.config.model.loss == "lovasz":
            seg_loss = MultiClassLovaszSoftmaxLoss()
        else:
            seg_loss = 'categorical_crossentropy'

        # define depth loss
        depth_loss = tf.keras.losses.MeanSquaredError()

        # composed loss
        losses = {
            "seg_out": seg_loss,
            "dep_out": depth_loss
        }

        loss_weights = {
            "seg_out": self.config.model.loss_weights.seg_loss,
            "dep_out": self.config.model.loss_weights.depth_loss
        }

        # define metrics
        metrics = {
            "seg_out": self._generate_metrics(),
            "dep_out": [tf.keras.metrics.RootMeanSquaredError()]
        }

        network.compile(
            loss=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics
        )

        return network

    def build_optimizer(self):
        # retrieve params from config
        momentum = self.config.model.optimizer.momentum
        initial_learning_rate = self.config.model.lr.initial
        power = self.config.model.lr.power
        cycle = self.config.model.lr.cycle

        # compute the total number of iterations through the epochs
        total_iterations = self.config.trainer.num_epochs * self.datagen.__len__()
        print('total iter', total_iterations)
        # poly learning rate policy
        poly_lr_policy = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_iterations,
            power=power,
            cycle=cycle
        )

        # SGD optimizer
        sgd = keras.optimizers.SGD(
            learning_rate=poly_lr_policy,
            momentum=momentum
        )
        return sgd

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

    def decoder(self, backbone, skips):
        stage5 = backbone.output  # resolution 1/32
        stage4 = skips[0]  # resolution 1/16
        stage3 = skips[1]  # resolution 1/8
        stage2 = skips[2]  # resolution 1/4

        # Pyramid pooling module for stage 5 tensor
        stage5 = ppm_block(stage5, (1, 2, 3, 6), 128, 1024)

        # channel controllers
        skip4 = conv2d(stage4, 512, 1, 1, kernel_size=1,
                       use_relu=True, name="decoder_skip4")
        skip3 = conv2d(stage3, 256, 1, 1, kernel_size=1,
                       use_relu=True, name="decoder_skip3")
        skip2 = conv2d(stage2, 128, 1, 1, kernel_size=1,
                       use_relu=True, name="decoder_skip2")

        # fusion nodes
        fusion4 = fusion_node(stage5, skip4, name="decoder_fusion4")
        fusion3 = fusion_node(fusion4, skip3, name="decoder_fusion3")
        fusion2 = fusion_node(fusion3, skip2, name="decoder_fusion2")
        print(fusion2.shape)

        # fusion nodes merging
        merge = merge_block(fusion4, fusion3, fusion2,
                            out_channels=64)
        print(merge.shape)
        return merge

    def segmentation_head(self, features):
        features = conv2d(features, 64, 1, 1, kernel_size=3,
                          use_relu=True, name="segmentation_head_3x3")
        features = conv2d(features, self.config.model.classes,
                          1, 1, kernel_size=1, use_relu=True, name="segmentation_head_1x1")

        upsampled = resize_img(
            features, self.config.model.height, self.config.model.width)

        segmentation_mask = Activation(
            'softmax', name='seg_out')(upsampled)
        return segmentation_mask

    def depth_head(self, features):
        features = conv2d(features, 64, 1, 1, kernel_size=3,
                          use_relu=True, name="depth_head_3x3")
        features = conv2d(features, 1,
                          1, 1, kernel_size=1, use_relu=True, name="depth_head_1x1")

        upsampled = resize_img(
            features, self.config.model.height, self.config.model.width, name="dep_out")

        return upsampled

# Layer functions


def merge_block(fusion4, fusion3, fusion2, out_channels):
    # get fusion2 channels number
    inter_channels = K.int_shape(fusion2)[-1]

    # fusion3 upsampling
    fusion3 = conv2d(fusion3, inter_channels, 1, 1,
                     kernel_size=1, use_relu=True)
    fusion3 = UpSampling2D(size=(2, 2))(fusion3)

    # fusion4 upsampling
    fusion4 = conv2d(fusion4, inter_channels, 1, 1,
                     kernel_size=1, use_relu=True)
    fusion4 = UpSampling2D(size=(4, 4))(fusion4)

    # addition
    merge = Add()([
        fusion4,
        fusion3,
        fusion2
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
        x = conv2d(x, inter_channels, 1, 1, kernel_size=1, use_relu=True)
        x = Lambda(lambda x: tf.image.resize(x, (H, W)))(x)
        concat.append(x)
    x = concatenate(concat)
    x = conv2d(x, out_channels, 1, 1, 3, use_relu=True)
    return x


def fusion_node(low_res, high_res, name=None):
    # define names
    if name is not None:
        low_conv_name = name + "_low"
        low_up_name = name + "_low_upsample"
        fus_name = name + "_concat"
        fus_conv_name = name + "_final"
    else:
        low_conv_name = low_up_name = fus_name = fus_conv_name = None

    # channel depth of output is high_res feature map depth
    out_channels = K.int_shape(high_res)[-1]

    # upsample the deepest, low-resolution feature map
    low_res = conv2d(low_res, out_channels, 1, 1, kernel_size=1,
                     use_relu=True, name=low_conv_name)
    low_res = UpSampling2D(size=(2, 2), name=low_up_name)(low_res)

    # fusion of two tensors
    fusion = concatenate([low_res, high_res], name=fus_name)
    fusion = conv2d(fusion, out_channels, 1, 1, kernel_size=3,
                    use_relu=True, name=fus_conv_name)

    return fusion


def conv2d(input, filters, stride, n, kernel_size=3, use_relu=True, name=None):
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

        if use_relu:
            x = Activation('relu', name=relu_name)(x)
    return x
