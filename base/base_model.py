from metrics.iou import build_iou_for, mean_iou
import tensorflow as tf
from tensorflow import keras


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError

    def _generate_metrics_sun_rgbd(self):
        """Generates the list of metrics to evaluate with SUN RGB-D.
        """
        metrics = self._generate_metrics(n_classes=38)
        ious = build_iou_for(
            label=range(38),
            name=[
                "background",  # 0
                "wall",  # 1
                "floor",  # 2
                "cabinet",  # 3
                "bed",  # 4
                "chair",  # 5
                "sofa",  # 6
                "table",  # 7
                "door",  # 8
                "window",  # 9
                "bookshelf",  # 10
                "picture",  # 11
                "counter",  # 12
                "blinds",  # 13
                "desk",  # 14
                "shelves",  # 15
                "curtain",  # 16
                "dresser",  # 17
                "pillow",  # 18
                "mirror",  # 19
                "floor_mat",  # 20
                "clothes",  # 21
                "ceiling",  # 22
                "books",  # 23
                "fridge",  # 24
                "tv",  # 25
                "paper",  # 26
                "towel",  # 27
                "shower_curtain",  # 28
                "box",  # 29
                "whiteboard",  # 30
                "person",  # 31
                "night_stand",  # 32
                "toilet",  # 33
                "sink",  # 34
                "lamp",  # 35
                "bathtub",  # 36
                "bag"  # 37
            ]
        )
        metrics.append(mean_iou)  # general mIoU
        metrics.append(keras.metrics.MeanIoU(num_classes=38))
        metrics.append('acc')
        return metrics

    def _nyu_v2_metrics(self):
        """Generates the metrics for the NYU-v2 dataset
        """
        metrics = self._generate_metrics(n_classes=40)
        # TODO
        return metrics

    def _generate_metrics(self, n_classes):
        metrics = []
        metrics.append(mean_iou)  # general mIoU
        metrics.append(keras.metrics.MeanIoU(num_classes=n_classes))
        metrics.append('acc')
        return metrics
