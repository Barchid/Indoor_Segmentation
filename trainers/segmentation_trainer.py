from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class SegmentationTrainer():
    def __init__(self, model, data_generator, config):
        """
        :param model: the compiled model to use
        :param data_generator: the data_generator to use (must inherit keras.utils.Sequence)
        """
        super(SegmentationTrainer, self).__init__(
            model, data_generator, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.mious = []  # table of mIoU metrics to use
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}-{loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):
        history = self.model.fit_generator(
            generator=self.data_generator,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks,
            use_multiprocessing=True if hasattr(
                self.config.trainer, 'workers') else False,
            workers=1 if not hasattr(
                self.config.trainer, 'workers') else self.config.trainer.workers
        )
