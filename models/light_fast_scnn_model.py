from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from layers.utils_layers import resize_img

# define conv block


def conv2d(input_map, c, s, n, kernel_size=3, use_relu=True):
    x = input_map
    for i in range(n):
        x = Conv2D(c, (kernel_size, kernel_size),
                   strides=(s, s), padding="same")(x)
        x = BatchNormalization()(x)
        if use_relu:
            x = keras.activations.relu(x)
    return x

# define DSconv block


def ds_conv2d(input_map, c, s, n, kernel_size=3):
    x = input_map
    for i in range(n):
        x = SeparableConv2D(c, (kernel_size, kernel_size),
                            strides=(s, s), padding="same")(x)
        x = BatchNormalization()(x)
        x = keras.activations.relu(x)
    return x

# define bottleneck function


def res_block(input_map, t, c, s, use_add=False):
    """residual connection block from original paper"""
    tc = t * keras.backend.int_shape(input_map)[-1]
    x = input_map

    # 1) Conv2D
    x = conv2d(x, tc, 1, 1, kernel_size=1)

    # 2) DWConv
    x = DepthwiseConv2D((3, 3), strides=(s, s), padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)

    # 3) last Conv2D without non linear activation fonction
    x = conv2d(x, c, 1, 1, kernel_size=1, use_relu=False)

    # skip connection
    if use_add:
        x = add([x, input_map])

    return x


def bottleneck(input_map, n, t, c, s):
    """Bottleneck block"""
    x = res_block(input_map, t, c, s, use_add=False)

    for i in range(1, n):
        x = res_block(x, t, c, 1, use_add=True)  # strides = 1 here

    return x

# PPM method


def ppm_block(inputs, bin_sizes, H, W, c):
    concat = [inputs]

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(W//bin_size, H//bin_size),
                             strides=(W//bin_size, H//bin_size))(inputs)
        x = Conv2D(c//len(bin_sizes), (3, 3),
                   strides=(2, 2), padding="same")(x)
        x = Lambda(lambda x: tf.image.resize(x, (W, H)))(x)
        concat.append(x)
    return concatenate(concat)

# FFM method


def ffm_block(low_resolution, high_resolution, c):
    # for higher res features
    high = conv2d(high_resolution, c, 1, 1, kernel_size=1, use_relu=False)

    # for lower res features
    low = UpSampling2D(size=(4, 4))(low_resolution)

    low = DepthwiseConv2D((3, 3), strides=(
        1, 1), padding="same", dilation_rate=(4, 4))(low)
    low = BatchNormalization()(low)
    low = keras.activations.relu(low)

    low = conv2d(low, c, 1, 1, kernel_size=1, use_relu=False)

    # resize feature if needed
    high_shape = keras.backend.int_shape(high)
    low_shape = keras.backend.int_shape(low)
    if(high_shape[1] != low_shape[1] or high_shape[2] != low_shape[2]):
        high = resize_img(
            high, low_shape[1], low_shape[2])

    # "add" fusion of both low and high features
    fusion = add([high, low])
    fusion = BatchNormalization()(fusion)
    fusion = keras.activations.relu(fusion)
    return fusion


class LightFastScnnModel(BaseModel):
    def __init__(self, config):
        super(LightFastScnnModel, self).__init__(config)

    def build_model(self):
        # input layer
        input_layer = Input(shape=(self.config.model.height,
                                   self.config.model.width, 3), name="input_layer")
        print(input_layer.shape)

        # Learning to down-sample module
        print("LEARNING TO DOWNSAMPLE")
        ltd1 = conv2d(input_layer, 32, 2, 1)
        print(ltd1.shape)

        ltd2 = ds_conv2d(ltd1, 48, 2, 1)
        print(ltd2.shape)

        ltd3 = ds_conv2d(ltd2, 64, 2, 1)
        print(ltd3.shape)

        print("GLOBAL FEATURE EXTRACTOR")
        # Adding the res blocks
        gfe1 = bottleneck(ltd3, 3, 6, 64, 2)
        print(gfe1.shape)

        gfe2 = bottleneck(gfe1, 3, 6, 96, 2)
        print(gfe2.shape)

        gfe3 = bottleneck(gfe2, 3, 6, 128, 1)
        print(gfe3.shape)

        # adding the PPM module into layers
        gfe4 = ppm_block(gfe3, [2, 4, 6], 23, 17, 128)
        print(gfe4.shape)

        ffm = ffm_block(gfe4, ltd3, 128)
        print(ffm.shape)

        print("CLASSIFIER")
        # classifier block method
        class1 = ds_conv2d(ffm, 128, 1, 2, kernel_size=3)
        print(class1.shape)

        class2 = conv2d(class1, self.config.model.classes,
                        1, 1, kernel_size=3, use_relu=True)

        class3 = UpSampling2D(size=(8, 8))(class2)

        class3_shape = keras.backend.int_shape(class3)
        if(class3_shape[1] != self.config.model.height or class3_shape[2] != self.config.model.width):
            class3 = resize_img(
                class3, self.config.model.height, self.config.model.width)

        prediction = keras.activations.softmax(class3)
        print(prediction.shape)

        fast_scnn = keras.Model(
            inputs=input_layer, outputs=prediction, name="FAST_SCNN")

        fast_scnn.summary()
        optimizer = self.build_optimizer()
        metrics = self.build_metrics_SUN()
        fast_scnn.compile(loss='categorical_crossentropy',
                          optimizer=optimizer, metrics=metrics)

        return fast_scnn
