from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.get_value(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
