import csv

from numpy import array


def predict(model, epoch):
    n_input = 12
    x_input = array([[0.7310585786300049, 0.5,
                      0.5, 0.5,
                      0.4789868468433692, 0.5758068303028442,
                      0.5, 0.5,
                      0.2857944826442093, 0.21322690668855132,
                      0.5, 0.5]])
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
