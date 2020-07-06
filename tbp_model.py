import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

from tbp_predict import predict

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

        if i != 0 and (i == 390 or i % 390 == 0):
            continue
        else:
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

    return array(X), array(y)


test_frame = pandas.read_csv("test_data.csv", index_col=None, header=0).to_numpy()
train_frame = pandas.read_csv("train_data.csv", index_col=None, header=0).to_numpy()

# choose a number of time steps
n_steps = 1

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


class PredictionCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predict(self.model, epoch)


model.fit(
    X_train,
    y_train,
    batch_size=5000,
    epochs=500,
    verbose=2,
    validation_data=(X_test, y_test),
    callbacks=[PredictionCallback()]
)

model.save("tbp-july-6-06_01")
