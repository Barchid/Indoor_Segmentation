from models.fpn_depth import FpnDepth
from utils import factory
from data_generators.joint_data_generator import JointDataGenerator
from utils.config import process_config
import cv2
import numpy as np
from utils.utils import get_args
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def create_visualization(pred_dep, gt_dep, pred_seg, gt_seg, img):
    pred_dep = cv2.cvtColor(pred_dep, cv2.COLOR_GRAY2BGR)
    pred_dep = (pred_dep*255).astype(np.uint8)

    gt_dep = cv2.cvtColor(gt_dep, cv2.COLOR_GRAY2BGR)
    gt_dep = (gt_dep*255).astype(np.uint8)

    # dep_heatmap = dep_heatmap.astype(np.float32)/255.
    # dep_overlay = dep_overlay.astype(np.float32)/255.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)

    # generate segmaps
    pred_seg = np.argmax(pred_seg, axis=-1).astype(np.uint8)
    gt_seg = np.argmax(gt_seg, axis=-1).astype(np.uint8)
    pred_seg = SegmentationMapsOnImage(pred_seg, shape=img.shape)
    gt_seg = SegmentationMapsOnImage(gt_seg, shape=img.shape)

    grid_image = ia.draw_grid([
        img,
        pred_seg.draw_on_image(img)[0],
        gt_seg.draw_on_image(img)[0],
        pred_dep,
        gt_dep
        # dep_heatmap,
        # dep_overlay,
    ], cols=5)
    return grid_image


def visualize_results(model, config, datagen):
    """Visualization of predictions
    """

    for i in range(len(datagen)):
        print('Processing sample n°', i, '...')
        # IF there is no depth data
        X, [Y, Z] = datagen[i]
        pred_seg, pred_dep = model.predict(X, batch_size=1, verbose=1)
        pred_dep = np.clip(pred_dep[0], 0., 1.)
        pred_seg = pred_seg[0]

        # depth_gradcam = DepthGradCam(model, depth_range=(
        #     config.validation.depth_range.start, config.validation.depth_range.end),
        #     layer_name=config.validation.layer_name)

        # dep_heatmap = depth_gradcam.compute_heatmap(X)
        # dep_heatmap, overlay = depth_gradcam.overlay_heatmap(
        #     dep_heatmap, X[0].copy())

        visualization = create_visualization(
            pred_dep, Z[0], pred_seg, Y[0], X[0])

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
    datagen = JointDataGenerator(config, is_training_set=False)

    network = factory.create(config.model.class_name)(config, datagen)

    # Load weight file in model
    load_weights(network.model, config)

    visualize_results(network.model, config, datagen)


if __name__ == "__main__":
    main()
