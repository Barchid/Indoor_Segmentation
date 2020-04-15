class BaseTrain(object):
    def __init__(self, model, data_generator, config, validation_generator=None):
        self.model = model
        self.data_generator = data_generator
        self.config = config
        self.validation_generator = validation_generator

    def train(self):
        raise NotImplementedError
