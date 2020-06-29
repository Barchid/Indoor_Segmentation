import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K


def res_block(input, out_channels, downsample_start=False, use_skip_conv=False):
    """Implementation of residual block
    :param downsample_start: flag that indicates whether the first conv layer must be downsampled with a stride 2
    """
    res = input
    x = input
    first_stride = (2, 2) if downsample_start else (1, 1)

    # first 3x3 conv
    x = Conv2D(out_channels, (3, 3), strides=first_stride,
               padding="same")(input)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)

    # second 3x3 convolution
    x = Conv2D(out_channels, (3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    # skip connection of input to reduce the resolution (needed to create the residual connection)
    if downsample_start:
        res = Conv2D(out_channels, (1, 1), strides=(2, 2), padding="same")(res)
        res = BatchNormalization()(res)

    # use skip connection of input to only change the channel depth
    if use_skip_conv:
        res = Conv2D(out_channels, (1, 1), strides=(1, 1), padding="same")(res)
        res = BatchNormalization()(res)

    res = add([res, x])
    res = keras.activations.relu(res)

    return res


def res_stage(input, out_channels, blocks, downsample_start=True):
    """Performs several sequential residual blocks and ends up with a downsampling operation
    :param blocks: the number of residual blocks in stage
    :param downsample_start: flag that indicates whether the first res block will downsample the resolution with stride 2
    """
    # first block is the strided one
    stage = res_block(input, out_channels, downsample_start=downsample_start)

    # next blocks are non-strided
    for i in range(1, blocks):
        stage = res_block(stage, out_channels)

    return stage


def DF1(input_tensor, weights='imagenet', pooling=False, include_top=False):
    """Builds the DF1 backbone network (without the flattened part)
    TODO: implement integration for all parameters
    """
    input = input_tensor
    print("INPUT")
    print(input.shape)

    # stage 1
    stage1 = Conv2D(32, (3, 3), strides=(2, 2), padding="same",
                    name="stage1_conv1")(input)
    stage1 = BatchNormalization(name="stage1_conv1_bn")(stage1)
    stage1 = keras.activations.relu(stage1)
    print("STAGE 1")
    print(stage1.shape)

    # stage 2
    stage2 = Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                    name="stage2_conv1")(stage1)
    stage2 = BatchNormalization(name="stage2_conv1_bn")(stage2)
    stage2 = keras.activations.relu(stage2)
    print("STAGE 2")
    print(stage2.shape)

    # stage 3
    stage3 = res_stage(stage2, 64, 3)
    print("STAGE 3")
    print(stage3.shape)

    # stage 4
    stage4 = res_stage(stage3, 128, 3)
    print("STAGE 4")
    print(stage4.shape)

    # stage 5
    stage5 = res_stage(stage4, 256, 3)
    print("STAGE 5")
    print(stage5.shape)
    stage5 = res_block(stage5, 512, downsample_start=False, use_skip_conv=True)
    print(stage5.shape)
    df1 = keras.Model(inputs=input, outputs=stage5, name="DF1")
    return df1


def DF2(input_tensor, weights='imagenet', pooling=False, include_top=False):
    print("INPUT")
    input = input_tensor
    print(input.shape)

    # stage 1
    print("STAGE 1")
    stage1 = Conv2D(32, (3, 3), strides=(2, 2), padding="same",
                    name="stage1_conv1")(input)
    stage1 = BatchNormalization(name="stage1_conv1_bn")(stage1)
    stage1 = keras.activations.relu(stage1)
    print(stage1.shape)

    # stage 2
    print("STAGE 2")
    stage2 = Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                    name="stage2_conv1")(stage1)
    stage2 = BatchNormalization(name="stage2_conv1_bn")(stage2)
    stage2 = keras.activations.relu(stage2)
    print(stage2.shape)

    # stage 3
    print("STAGE 3")
    stage3 = res_stage(stage2, 64, 2)
    print(stage3.shape)
    stage3 = res_block(stage3, 128, downsample_start=False, use_skip_conv=True)
    print(stage3.shape)

    # stage 4
    print("STAGE 4")
    stage4 = res_stage(stage3, 128, 10)
    print(stage4.shape)
    stage4 = res_block(stage4, 256, downsample_start=False, use_skip_conv=True)
    print(stage4.shape)

    # stage 5
    print("STAGE 5")
    stage5 = res_stage(stage4, 256, 4)
    print(stage5.shape)
    stage5 = res_block(stage5, 512, downsample_start=False, use_skip_conv=True)
    stage5 = res_block(stage5, 512, downsample_start=False,
                       use_skip_conv=False)
    print(stage5.shape)

    df2 = keras.Model(inputs=input, outputs=stage5, name="DF2")
    return df2
