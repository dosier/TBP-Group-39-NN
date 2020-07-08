import csv

import numpy
from numpy import array


def predict(model, epoch):
    """
    Predicts a time series ranging from 0 to 3.9 seconds
    based on the initial conditions.

    Parameters
    ----------
    model : a keras model
    epoch : the current training epoch
    """

    n_input = 12
    x_input = array([[0.7310503405102471, 0.4999957996600001, 0.4980974039329515, 0.49923701259223147,
                      0.32051074910054406, 0.45810361982231246, 0.5032794754712855, 0.496923196337204,
                      0.43818473377629197, 0.3032215439837178, 0.498623086980649, 0.5038397545143354]])
    x_input = x_input.reshape((1, n_input))

    with open("predictions/epoch-" + str(epoch) + ".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow("x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3")
        t = 0.0
        dt = 0.01
        while t < 3.9:
            prediction = model.predict(x_input, verbose=0)
            writer.writerow(prediction[0])
            x_input = prediction
            t += dt


def predict_multi(model, epoch, x_input):
    """
    Predicts a time series ranging from 0 to 3.9 seconds
    based on the initial conditions.

    Parameters
    ----------
    model : a keras model
    epoch : the current training epoch
    x_input : the input used as initial conditions for the model (multiple)
    """

    ip = numpy.copy(x_input)
    with open("predictions/epoch-" + str(epoch) + ".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow("x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3")

        t = 0.0
        dt = 0.01
        split = numpy.array_split(ip, len(x_input)/12)
        for a in split:
            writer.writerow(a)
            t += dt

        while t < 3.9:
            ip = numpy.array([ip])
            # print(ip.shape)
            prediction = model.predict(ip, verbose=0)[0]
            writer.writerow(prediction)
            for i in range(len(split)-1):
                split[i] = split[i+1]
            split[len(split)-1] = prediction
            ip = numpy.concatenate(split)
            t+=dt
