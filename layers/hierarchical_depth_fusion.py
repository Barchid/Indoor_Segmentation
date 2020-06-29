import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from layers.bi_fpn import BiFpnLayer


def HierarchicalDepthFusion(P2, P3, P4, P5, u=2, v=1, filters=64, conv_input=True, name="HDF_"):
    # segmentation submodule
    P2_seg, P3_seg, P4_seg, P5_seg = BiFpnLayer(
        P2, P3, P4, P5, filters=filters, conv_input=conv_input, name=name + "BiFpn_seg_0_")
    for i in range(u-1):
        P2_seg, P3_seg, P4_seg, P5_seg = BiFpnLayer(
            P2_seg, P3_seg, P4_seg, P5_seg, filters=filters, conv_input=False, name=name + "BiFpn_seg_" + str(i+1) + "_")

    # depth submodule
    P2_dep, P3_dep, P4_dep, P5_dep = BiFpnLayer(
        P2, P3, P4, P5, filters=filters, conv_input=conv_input, name=name + "BiFpn_dep_0_")
    for i in range(u-1):
        P2_dep, P3_dep, P4_dep, P5_dep = BiFpnLayer(
            P2_dep, P3_dep, P4_dep, P5_dep, filters=filters, conv_input=False, name=name + "BiFpn_dep_" + str(i+1) + "_")

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


def conv2d(input, filters, stride, n, kernel_size=3, name=None):
    x = input
    for i in range(n):
        # define names for layers
        if name is not None:
            conv_name = name + "_conv_" + str(i)
            bn_name = name + "_bn_" + str(i)
            swish_name = name + "_swish_" + str(i)
        else:
            conv_name = bn_name = swish_name = None

        x = Conv2D(filters, (kernel_size, kernel_size), strides=(
            stride, stride), padding="same", name=conv_name)(x)

        x = BatchNormalization(name=bn_name)(x)

        x = Activation(lambda a: tf.nn.swish(a), name=swish_name)(x)
    return x
