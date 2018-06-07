'''
    Trains a DNN on the MNIST dataset.
    loss: 0.0107 - acc: 0.9967 - val_loss: 0.0994 - val_acc: 0.9810
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense
from utils import *

batch_size = 64
num_classes = 10
epochs = 20

# Load and preparing the train and test data

x_train, y_train, x_test, y_test = prepare_data(num_classes)

# Creating the model

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784, )))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(x_train,
#           y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           callbacks=[remote])

fit_model(model, x_train, y_train, x_test, y_test, batch_size, epochs)
