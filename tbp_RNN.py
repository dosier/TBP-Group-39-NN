
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tensorflow import keras
import glob
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    l = len(sequences)
    for i in range(l):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > l - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def read_data(path):
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, nrows=392)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True).to_numpy()


test_frame = read_data(path='C:\\Users\\gasto\\Documents\\NeuralNetworks\\TBP-Group-39\\Test_data')
train_frame = read_data(path='C:\\Users\\gasto\\Documents\\NeuralNetworks\\TBP-Group-39\\train_data')

# choose a number of time steps
n_steps = 1

# convert into input/output
X_train, y_train = split_sequences(train_frame, n_steps)
X_test, y_test = split_sequences(test_frame, n_steps)

# flatten input
n_input = X_train.shape[1] * X_train.shape[2]

n_output = y_train.shape[1]

logdir = "logs\scalars" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.

model = keras.Sequential()
model.add(keras.layers.LSTM(128, return_sequences = True, input_shape=(X_train.shape[0], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(128, return_sequences = True))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(128, return_sequences = True))
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(n_output))

print(model.summary())
model.compile(optimizer='adam', loss='mse')
y_train = y_train.reshape(1, X_train.shape[0], X_train.shape[2])
X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[2])
print(y_train.shape)
print(X_train.shape)
print("voor fit")
model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=100,
    verbose=2,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback]
)

model.save("tbp-rnn-overload")







# model.add(keras.layers.Embedding(input_dim=n_input, output_dim=64))
# print(len(X_train))
# print(y_train.shape)
# # Add a LSTM layer with 128 internal units.
# model.add(keras.layers.LSTM(units = 128, return_sequences = True, input_shape=(n_steps, 12)))
# model.add(keras.layers.Dropout(0.2))
#
# model.add(keras.layers.LSTM(units = 128, return_sequences = True))
# model.add(keras.layers.Dropout(0.2))
#
# model.add(keras.layers.LSTM(units = 128, return_sequences = True))
# model.add(keras.layers.Dropout(0.2))
#
# # Add a Dense layer with 10 units.
# model.add(keras.layers.Dense(n_output))
#
# model.compile(optimizer = 'adam', loss = "mean_squared_error")
#
# model.fit(
#     X_train,
#     y_train,
#     epochs=100,
#     batch_size= 32,
#     validation_data=(X_test, y_test),
#     callbacks=[tensorboard_callback])
#
# model.save("tbp-rnn-overload")
# # print(model.summary())
# # print(dataset_train)



















































# def rnn_cell_forward(xt, a_prev, parameters):
#     Wax = parameters["Wax"]
#     Waa = parameters["Waa"]
#     Wya = parameters["Wya"]
#     ba = parameters["ba"]
#     by = parameters["by"]
#
#     a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba);
#     yt_pred = softmax(np.dot(Wya, a_next) + by);
#
#     cache = (a_next, a_prev, xt, parameters)
#
#     return a_next, yt_pred, cache
#
#
# def rnn_forward_pass(x, a0, parameters):
#     caches = []
#
#     _, m, T_x = x.shape
#     n_y, n_a = parameters["Wya"].shape
#     a = np.zeros((n_a, m, T_x))
#     y_pred = np.zeros((n_y, m, T_x))
#     a_next = a0
#     for t in range(T_x):
#         a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
#         a[:, :, t] = a_next
#         y_pred[:, :, t] = yt_pred
#     caches.append(cache)
#     caches = (caches, x)
#     return a, y_pred, caches
#
# def rnn_cell_backward(da_next, cache):
#     (a_next, a_prev, xt, parameters) = cache
#     Wax = parameters["Wax"]
#     Waa = parameters["Waa"]
#     Wya= parameters["Wya"]
#     ba= parameters["ba"]
#     by= parameters["by"]
#     dtanh = (1 - a_next * a_next) * da_next
#     dWax = np.dot(dtanh, xt.T)
#     dxt = np.dot(Wax.T, dtanh)
#     dWaa = np.dot(dtanh, a_prev.T)
#     da_prev = np.dot(Waa.T, dtanh)
#     dba = np.sum(dtanh, keepdims=True, axis=-1)
#     gradients = {"dxt": dxt, "da_prev": da_prev,"dWax": dWax, "dWaa": dWaa, "dba": dba}
#     return gradients
#
# def rnn_backward(da, caches):
#     caches, x = caches
#     a1, a0, x1, parameters = caches[0]
#     n_a, m, T_x = da.shape
#     n_x, m = x1.shape
#     dx = np.zeros((n_x, m, T_x))
#     dWax = np.zeros((parameters['Wax'].shape))
#     dWaa = np.zeros((parameters['Waa'].shape))
#     dba = np.zeros((parameters['ba'].shape))
#     da0 = np.zeros(a0.shape)
#     da_prevt = np.zeros((n_a, m))
#
#
#     for t in reversed(range(T_x)):
#         gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
#         dxt, da_prevt, dWaxt, dWaat, dbat = gradients['dxt'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']
#         dWax += dWaxt
#         dWaa += dWaat
#         dba += dbat
#         dx[:, :, t] = dxt
#         da0 = da_prevt
#     gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}
#     return gradients