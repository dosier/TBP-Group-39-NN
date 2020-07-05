import csv

from numpy import array
from tensorflow import keras

model = keras.models.load_model("tbp-rnn-overload")

n_input = 12
x_input = array([[1, 0, 0, 0, -4.342523e-01, 6.383026e-02, 0, 0, -5.657477e-01, -1.063830e+00, 0, 0]])
x_input = x_input.reshape((1, x_input.shape[0], n_input))
print(x_input.shape)

with open("out2.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow("x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3")
    t = 0.0
    dt = 0.01
    while t < 3.9:
        input = model.predict(x_input, verbose=2)
        writer.writerow(input[0])
        x_input = input
        t += dt
