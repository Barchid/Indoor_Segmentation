import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

# greatly inspired from https://github.com/xuannianz/EfficientDet/blob/master/layers.py
# from the paper "EfficientDet: Scalable and Efficient Object Detection"


class FastNormalizedFusion(keras.layers.Layer):
    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(
            name=self.name,
            shape=(num_in, ),
            initializer=keras.initializers.constant(1 / num_in),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, **kwargs):
        # ensure non-zero
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i]
                           for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + keras.backend.epsilon())

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def BiFpnLayer(P2, P3, P4, P5, filters=64, conv_input=True, has_bottomup=True, name="BiFpn_"):
    """Implementation of BiFPN layer
    :param conv_input: True if the input features (P1,P2,etc) must be convoluted before applying BiFPN layer 
    :param has_bottomup: indicates whether there is a bottom-up pathway
    """
    P2_in = P2
    P3_in = P3
    P4_in = P4
    P5_in = P5

    # apply 1x1 conv if needed
    if conv_input:
        P2_in = conv2d(P2_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P2_in_conv_channels")
        P3_in = conv2d(P3_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P3_in_conv_channels")
        P4_in = conv2d(P4_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P4_in_conv_channels")
        P5_in = conv2d(P5_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P5_in_conv_channels")

    # compute P4_td
    P5_up = UpSampling2D(name=name + "P5_in_up")(P5_in)
    P4_td = FastNormalizedFusion(name=name + "P4_td_add")([P4_in, P5_up])
    P4_td = conv2d(P4_td, filters=filters, stride=1,
                   n=1, kernel_size=3, name=name + "P4_td_conv")

    # compute P3_td
    P4_td_up = UpSampling2D(name=name + "P4_td_up")(P4_td)
    P3_td = FastNormalizedFusion(name=name + "P3_td_add")([P3_in, P4_td_up])
    P3_td = conv2d(P3_td, filters=filters, stride=1,
                   n=1, kernel_size=3, name=name + "P3_td_conv")

    # compute P2_out
    P3_td_up = UpSampling2D(name=name + "P3_td_up")(P3_td)
    P2_out = FastNormalizedFusion(name=name + "P2_out_add")([P2_in, P3_td_up])
    P2_out = conv2d(P2_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P2_out_conv")

    if not has_bottomup:
        return P2_out

    # compute P3_out
    P2_out_down = MaxPooling2D(
        padding="same", name=name + "P2_out_down")(P2_out)
    P3_out = FastNormalizedFusion(name=name + "P3_out_add")(
        [P3_in, P3_td, P2_out_down])
    P3_out = conv2d(P3_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P3_out_conv")

    # compute P4_out
    P3_out_down = MaxPooling2D(
        padding="same", name=name + "P3_out_down")(P3_out)
    P4_out = FastNormalizedFusion(name=name + "P4_out_add")(
        [P4_in, P4_td, P3_out_down])
    P4_out = conv2d(P4_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P4_out_conv")

    # compute P5_out
    P4_out_down = MaxPooling2D(
        padding="same", name=name + "P4_out_down")(P4_out)
    P5_out = FastNormalizedFusion(
        name=name + "P5_out_add")([P4_out_down, P5_in])
    P5_out = conv2d(P5_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P5_out_conv")

    return P2_out, P3_out, P4_out, P5_out


def BiFpnDeepLayer(P1, P2, P3, P4, P5, filters=64, conv_input=True, has_bottomup=True, name="BiFpn_"):
    """Implementation of BiFPN layer
    :param conv_input: True if the input features (P1,P2,etc) must be convoluted before applying BiFPN layer 
    :param has_bottomup: indicates whether there is a bottom-up pathway
    """
    P1_in = P1
    P2_in = P2
    P3_in = P3
    P4_in = P4
    P5_in = P5

    # apply 1x1 conv if needed
    if conv_input:
        P1_in = conv2d(P1_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P1_in_conv_channels")
        P2_in = conv2d(P2_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P2_in_conv_channels")
        P3_in = conv2d(P3_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P3_in_conv_channels")
        P4_in = conv2d(P4_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P4_in_conv_channels")
        P5_in = conv2d(P5_in, filters=filters, stride=1, n=1,
                       kernel_size=1, name=name + "P5_in_conv_channels")

    # compute P4_td
    P5_up = UpSampling2D(name=name + "P5_in_up")(P5_in)
    P4_td = FastNormalizedFusion(name=name + "P4_td_add")([P4_in, P5_up])
    P4_td = conv2d(P4_td, filters=filters, stride=1,
                   n=1, kernel_size=3, name=name + "P4_td_conv")

    # compute P3_td
    P4_td_up = UpSampling2D(name=name + "P4_td_up")(P4_td)
    P3_td = FastNormalizedFusion(name=name + "P3_td_add")([P3_in, P4_td_up])
    P3_td = conv2d(P3_td, filters=filters, stride=1,
                   n=1, kernel_size=3, name=name + "P3_td_conv")

    # compute P2_td
    P3_td_up = UpSampling2D(name=name + "P3_td_up")(P3_td)
    P2_td = FastNormalizedFusion(name=name + "P2_td_add")([P2_in, P3_td_up])
    P2_td = conv2d(P2_td, filters=filters, stride=1,
                   n=1, kernel_size=3, name=name + "P2_td_conv")

    # compute P1_out
    P2_td_up = UpSampling2D(name=name + "P2_td_up")(P2_td)
    P1_out = FastNormalizedFusion(name=name + "P1_out_add")([P1_in, P2_td_up])
    P1_out = conv2d(P1_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P1_out_conv")

    if not has_bottomup:
        return P1_out

    # compute P2_out
    P1_out_down = MaxPooling2D(
        padding="same", name=name + "P1_out_down")(P1_out)
    P2_out = FastNormalizedFusion(name=name + "P2_out_add")(
        [P2_in, P2_td, P1_out_down])
    P2_out = conv2d(P2_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P2_out_conv")

    # compute P3_out
    P2_out_down = MaxPooling2D(
        padding="same", name=name + "P2_out_down")(P2_out)
    P3_out = FastNormalizedFusion(name=name + "P3_out_add")(
        [P3_in, P3_td, P2_out_down])
    P3_out = conv2d(P3_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P3_out_conv")

    # compute P4_out
    P3_out_down = MaxPooling2D(
        padding="same", name=name + "P3_out_down")(P3_out)
    P4_out = FastNormalizedFusion(name=name + "P4_out_add")(
        [P4_in, P4_td, P3_out_down])
    P4_out = conv2d(P4_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P4_out_conv")

    # compute P5_out
    P4_out_down = MaxPooling2D(
        padding="same", name=name + "P4_out_down")(P4_out)
    P5_out = FastNormalizedFusion(
        name=name + "P5_out_add")([P4_out_down, P5_in])
    P5_out = conv2d(P5_out, filters=filters, stride=1,
                    n=1, kernel_size=3, name=name + "P5_out_conv")

    return P1_out, P2_out, P3_out, P4_out, P5_out


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
