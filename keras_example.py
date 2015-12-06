import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(output_dim=64, input_dim=100, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dense(output_dim=10,init='glorot_uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd')


if __name__ == "__main__":
    pass