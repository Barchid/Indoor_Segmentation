# Grad cam visualization as showed in https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


class GradCam:
    def __init__(self, model, class_id, layer_name=None):
        self.model = model
        self.class_id = class_id
        self.layer_name = layer_name

        if self.layer_name is None:
            self.layer_name = self.find_default_layer()

    def find_default_layer(self):
        """
        Tries to find the final conv layer (called if the layer_name parameter is not provided)
        """
        for layer in reversed(self.model.layers):
            print(layer.name)
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        """
        Computes the heatmap by grad CAM
        """

        print("image", image.shape)
        # construct the grad model
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        print("inputs grad model", grad_model.inputs)
        print("outputs grad model", grad_model.outputs)

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            print("cast image inputs", inputs.shape)
            (conv_outputs, predictions) = grad_model(inputs)
            print("outputs shapes", conv_outputs.shape, predictions.shape)
            loss = predictions[:, self.class_id]
            print("Loss", loss)

        grads = tape.gradient(loss, conv_outputs)
        print("grads", grads.shape)

        # compute guided gradients
        positive_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        positive_grads = tf.cast(grads > 0, "float32")
        guided_grads = positive_conv_outputs * positive_grads * grads
        print("grads guided", guided_grads.shape)

        # remove batch dimension (useless here because there is only 1 image)
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        # compute weights (see eq in paper)
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        print("weights", weights.shape)

        # compute grad CAM
        mul = tf.multiply(weights, conv_outputs)
        print("mul", mul.shape)
        cam = tf.math.reduce_sum(mul, axis=-1)
        print("cam", cam.shape)

        # image dimensions
        (W, H) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (W, H))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
