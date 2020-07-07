from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from layers.utils_layers import resize_img
from losses.focal_loss import CategoricalFocalLoss
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
}


class HdafNet(BaseModel):
    # Hierarchical Depth-Aware Fusion
    def __init__(self, config, datagen):
        super(HdafNet, self).__init__(config, datagen)

    def build_model(self):
        # backbone encoder
        backbone, skips = self.get_backbone()

        # joint decoder (resolution 1/4)
        seg_output, seg_tmps, dep_tmps = self.decoder(backbone, skips)

        # create outputs, losses & loss_weights to compile the model
        outputs = [seg_output]
        losses = {
            fix_output_name(seg_output.name): self.segmentation_loss()
        }
        loss_weights = {
            fix_output_name(seg_output.name): self.config.model.loss_weights.main_segmentation
        }
        # define metrics
        metrics = {
            fix_output_name(seg_output.name): self._generate_metrics()
        }
        # handle the auxiliary outputs in segmentation
        for seg_tmp in seg_tmps:
            outputs.append(seg_tmp)
            losses[fix_output_name(seg_tmp.name)] = self.segmentation_loss()
            loss_weights[fix_output_name(
                seg_tmp.name)] = self.config.model.loss_weights.deep_segmentation
            metrics[fix_output_name(seg_tmp.name)] = self._generate_metrics()

        # handle the auxiliary outputs in depth estimation
        for dep_tmp in dep_tmps:
            outputs.append(dep_tmp)
            losses[fix_output_name(dep_tmp.name)] = self.depth_loss()
            loss_weights[fix_output_name(
                dep_tmp.name)] = self.config.model.loss_weights.deep_depth
            metrics[fix_output_name(dep_tmp.name)] = [
                tf.keras.metrics.RootMeanSquaredError()]

        # the last depth output is the main depth output (so use the appropriate weight)
        loss_weights[fix_output_name(
            dep_tmps[-1].name)] = self.config.model.loss_weights.main_depth

        # create model
        network = keras.Model(
            inputs=backbone.input,
            outputs=outputs,
            name="HDAFNET"
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
        P5 = backbone.output  # resolution 1/32
        P4 = skips[0]  # resolution 1/16
        P3 = skips[1]  # resolution 1/8
        P2 = skips[2]  # resolution 1/4

        # Pyramid pooling module for stage 5 tensor
        P5 = ppm_block(P5, (1, 2, 3, 6), 128, 512)

        # define parameters of HDAF modules
        s = self.config.hdaf.s  # number of HDAF modules
        u = self.config.hdaf.u  # number of chained BiFPN blocks in depth & seg branches
        v = self.config.hdaf.v  # number of chained BiFPN blocks in fusion branch
        f = self.config.hdaf.f  # number of filters for an HDAF module
        seg_tmps, dep_tmps = [], []  # list of auxiliary outputs

        for i in range(s):
            # HDAF modules
            P2, P3, P4, P5, seg_tmp, dep_tmp = self.HierarchicalDepthAwareFusion(
                P2, P3, P4, P5, filters=f, u=u, v=v, name="HDF_" + str(i) + "_")
            seg_tmps.append(seg_tmp)
            dep_tmps.append(dep_tmp)

        # segmentation head
        seg_out = self.segmentation_head(P2, P3, P4, P5, name="main_seg_")
        return seg_out, seg_tmps, dep_tmps

    def HierarchicalDepthAwareFusion(self, P2, P3, P4, P5, u=1, v=1, filters=64, conv_input=True, name="HDF_"):
        # segmentation submodule
        P2_seg, P3_seg, P4_seg, P5_seg = BiFpnLayer(
            P2, P3, P4, P5, filters=filters, conv_input=conv_input, name=name + "BiFpn_seg_0_")
        for i in range(u-1):
            P2_seg, P3_seg, P4_seg, P5_seg = BiFpnLayer(
                P2_seg, P3_seg, P4_seg, P5_seg, filters=filters, conv_input=False, name=name + "BiFpn_seg_" + str(i+1) + "_")

        # deep supervision
        seg_tmp = self.segmentation_head(
            P2_seg, P3_seg, P4_seg, P5_seg, name=name + "seg_head_")

        # depth submodule
        P2_dep, P3_dep, P4_dep, P5_dep = BiFpnLayer(
            P2, P3, P4, P5, filters=filters, conv_input=conv_input, name=name + "BiFpn_dep_0_")
        for i in range(u-1):
            P2_dep, P3_dep, P4_dep, P5_dep = BiFpnLayer(
                P2_dep, P3_dep, P4_dep, P5_dep, filters=filters, conv_input=False, name=name + "BiFpn_dep_" + str(i+1) + "_")

        # deep supervision
        dep_tmp = self.depth_head(
            P2_dep, P3_dep, P4_dep, P5_dep, name=name+"dep_head_")

        # hierarchically concatenate depth and seg feature
        P2 = concatenate([P2_seg, P2_dep], name=name + "concatenate_P2")
        P3 = concatenate([P3_seg, P3_dep], name=name + "concatenate_P3")
        P4 = concatenate([P4_seg, P4_dep], name=name + "concatenate_P4")
        P5 = concatenate([P5_seg, P5_dep], name=name + "concatenate_P5")

        # hierarchical fusion
        P2, P3, P4, P5 = BiFpnLayer(
            P2, P3, P4, P5, filters=filters, conv_input=True, name=name + "BiFpn_fusion_0_")
        for i in range(v-1):
            P2, P3, P4, P5 = BiFpnLayer(
                P2, P3, P4, P5, filters=filters, conv_input=False, name=name + "BiFpn_fusion_" + str(i+1) + "_")

        return P2, P3, P4, P5, seg_tmp, dep_tmp

    def segmentation_head(self, P2, P3, P4, P5, name="seg_head_"):
        features = merge_block(P5, P4, P3, P2, 64, name=name+"merge_block_")
        features = conv2d(features, self.config.model.classes,
                          1, 1, kernel_size=1, name=name + "conv1x1")

        upsampled = resize_img(
            features, self.config.model.height, self.config.model.width)

        segmentation_mask = Activation(
            'softmax', name=name + "out")(upsampled)
        return segmentation_mask

    def depth_head(self, P2, P3, P4, P5, name="dep_head_"):
        features = merge_block(P5, P4, P3, P2, 64, name=name+"merge_block_")
        features = conv2d(features, 1,
                          1, 1, kernel_size=1, name=name + "conv1x1")

        upsampled = resize_img(
            features, self.config.model.height, self.config.model.width, name=name+"out")

        return upsampled

# Layer functions


def merge_block(P5, P4, P3, P2, out_channels, name="merge_block_"):
    # Upsamplings
    P3 = UpSampling2D(size=(2, 2), name=name+"P3_up")(P3)
    P4 = UpSampling2D(size=(4, 4), name=name+"P4_up")(P4)
    P5 = UpSampling2D(size=(8, 8), name=name+"P5_up")(P5)

    # addition
    merge = FastNormalizedFusion()([
        P2,
        P3,
        P4,
        P5
    ], name=name+"add")

    # last conv layer
    merge = conv2d(merge, out_channels, 1, 1,
                   kernel_size=3, name=name+"conv3x3")
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


def fix_output_name(name: str):
    """Removes the "Identity:0" of a tensor's name if it exists"""
    return name.replace("/Identity:0", "", 1)
