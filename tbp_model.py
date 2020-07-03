# multivariate output data prep
from datetime import datetime
import glob

import pandas
from numpy import array
# split a multivariate sequence into samples
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


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
        df = pandas.read_csv(filename, index_col=None, header=0, nrows=392)
        li.append(df)
    return pandas.concat(li, axis=0, ignore_index=True).to_numpy()


test_frame = read_data(path='test_data')
train_frame = read_data(path='train_data')

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

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)



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
model.compile(optimizer='adam', loss='mse')

model.fit(
    X_train,
    y_train,
    epochs=500,
    verbose=2,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback]
)

model.save("tbp-op-overload")

# demonstrate prediction
x_input = array([[1, 0, 0, 0, -1.256262e-01, 1.819678e-01, 0, 0, -8.743738e-01, -1.181968e+00, 0, 0]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=1)
print(yhat)

x_input = array([[9.999751e-01, 2.162537e-06, -5.117588e-03, 4.452208e-04, -3.818625e-01, 6.328183e-01, 3.629409e-03,
                  -3.613076e-03, -6.181126e-01, -1.632820e+00, 1.488179e-03, 3.167855e-03]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=2)
print(yhat)
yhat = model.predict(yhat, verbose=2)
print(yhat)
yhat = model.predict(yhat, verbose=2)
print(yhat)