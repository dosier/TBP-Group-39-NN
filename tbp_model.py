import os

import numpy

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras

from tbp_predict import predict_multi, predict

import pandas
from numpy import array
# split a multivariate sequence into samples
from keras.layers import Dense
from keras.models import Sequential


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    l = len(sequences)
    for i in range(l):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix + 1 >= l - 1:
            break

        if i != 0 and i % 390 == 0:
            continue
        else:
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

    return array(X), array(y)

test_frame = pandas.read_csv("test_data.csv", index_col=None, header=0).to_numpy()
train_frame = pandas.read_csv("train_data.csv", index_col=None, header=0).to_numpy()

# choose a number of time steps
n_steps = 10

# convert into input/output
X_train, y_train = split_sequences(train_frame, n_steps)
X_test, y_test = split_sequences(test_frame, n_steps)

# flatten input
n_input = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape((X_train.shape[0], n_input))
X_test = X_test.reshape((X_test.shape[0], n_input))

n_output = y_train.shape[1]

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(128, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mean_absolute_error')

print(model.summary())

name = "tbp-mlp-10-layers-128-units-multi-in-2"


class PredictionCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if n_steps == 1:
            predict(self.model, epoch + 1)
        else:
            predict_multi(self.model, epoch + 1, X_test[0])


history = model.fit(
    X_train,
    y_train,
    batch_size=3900,
    epochs=500,
    verbose=2,
    validation_data=(X_test, y_test),
    callbacks=[PredictionCallback()]
)

model.save(name + ".h5")
numpy.save(name+'_history.npy', history.history)