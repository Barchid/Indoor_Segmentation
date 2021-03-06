from models.fpn_depth import FpnDepth
from utils import factory
from data_generators.depth_data_generator import DepthDataGenerator
from utils.config import process_config
from utils.depth_grad_cam import DepthGradCam
import cv2
import numpy as np
from utils.utils import get_args
import imgaug as ia
import imgaug.augmenters as iaa


def create_visualization(pred, gt, img, heatmap, overlay):
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    heatmap = heatmap.astype(np.float32)/255.
    overlay = overlay.astype(np.float32)/255.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    grid_image = ia.draw_grid([
        img,
        pred,
        gt,
        heatmap,
        overlay
    ], cols=5)
    return grid_image


def visualize_results(model, config, datagen):
    """Visualization of predictions
    """

    for i in range(len(datagen)):
        print('Processing sample n°', i, '...')
        # IF there is no depth data
        X, Y = datagen[i]
        prediction = model.predict(X, batch_size=1, verbose=1)
        prediction = np.clip(prediction[0], 0., 1.)

        gradcam = DepthGradCam(model, depth_range=(
            config.validation.depth_range.start, config.validation.depth_range.end),
            layer_name=config.validation.layer_name)

        heatmap = gradcam.compute_heatmap(X)
        heatmap, overlay = gradcam.overlay_heatmap(heatmap, X[0].copy())

        visualization = create_visualization(
            prediction, Y[0], X[0], heatmap, overlay)

        # display
        cv2.imshow('Image - Ground truth - Prediction', visualization)
        if cv2.waitKey() == ord('a'):
            print('STOP')
            exit()


def load_weights(model, config):
    """Load weights file if required by configuration
    """
    if not hasattr(config, 'validation') or type(config.validation.weights_file) != str:
        return

    print('Load weight file : ', config.validation.weights_file)
    model.load_weights(config.validation.weights_file)


def main():
        # capture the config path from the run arguments
        # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Force some parameters from configuration
    config.generator.use_data_augmentation = False
    config.trainer.batch_size = 1

    # Data generator creation
    datagen = DepthDataGenerator(config, is_training_set=False)

    network = factory.create(config.model.class_name)(config, datagen)

    # Load weight file in model
    load_weights(network.model, config)

    visualize_results(network.model, config, datagen)


if __name__ == "__main__":
    main()
