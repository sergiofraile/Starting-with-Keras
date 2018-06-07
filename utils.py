import keras
from keras.datasets import mnist
from keras import callbacks


def prepare_data(num_classes):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('x_train.shape=', x_train.shape)
    print('y_train.shape=', y_train.shape)

    print('x_test.shape=', x_test.shape)
    print('y_test.shape=', y_test.shape)

    return x_train, y_train, x_test, y_test

def fit_model(m, kx_train, ky_train, kx_test, ky_test, batch_size=128, max_epochs=1000):

    checkpoint = callbacks.ModelCheckpoint(monitor='val_acc',
                                           filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
                                           save_best_only=True)

    early_stopping = callbacks.EarlyStopping(monitor='val_acc',
                                             min_delta=0.01,
                                             patience=10,
                                             verbose=1,
                                             mode='max')

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            factor=0.5,
                                            patience=10,
                                            min_lr=0.0001,
                                            verbose=1)

    m.fit(kx_train,
          ky_train,
          batch_size=batch_size,
          epochs=max_epochs,
          verbose=1,
          validation_data=(kx_test, ky_test),
          callbacks=[checkpoint, early_stopping, reduce_lr])
