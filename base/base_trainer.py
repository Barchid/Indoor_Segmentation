class BaseTrain(object):
    def __init__(self, model, data_generator, config):
        self.model = model
        self.data_generator = data_generator
        self.config = config

    def train(self):
        raise NotImplementedError
