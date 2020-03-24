from base.base_model import BaseModel
from tensorflow import keras
from tensorflow.keras.layers import *

class FastScnnModel(BaseModel):
    def __init__(self, config):
        super(FastScnnModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(
              loss='sparse_categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])
