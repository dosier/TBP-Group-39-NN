import pandas
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch
import numpy as np


def drange(start, end, increment, round_decimal_places=None):
    result = []
    if start < end:
        # Counting up, e.g. 0 to 0.4 in 0.1 increments.
        if increment < 0:
            raise Exception("Error: When counting up, increment must be positive.")
        while start <= end:
            result.append(start)
            start += increment
            if round_decimal_places is not None:
                start = round(start, round_decimal_places)
    else:
        # Counting down, e.g. 0 to -0.4 in -0.1 increments.
        if increment > 0:
            raise Exception("Error: When counting down, increment must be negative.")
        while start >= end:
            result.append(start)
            start += increment
            if round_decimal_places is not None:
                start = round(start, round_decimal_places)
    return result


data = pandas.read_csv("predictions/output_182_normalized.csv", index_col=None, header=0)

# data= np.array(data)
# save_data = []
# for j in range(len(data)):
#     for i in range(12):
#         save_data[j][i] = data[j][i]
#
# print(save_data)
timesteps = drange(0, 3.9, 0.01, round_decimal_places=3)
# data.insert(0, "timestep", timesteps)
print(data)
ax = data.plot(x="x1", y="y1")

x1 = data.at[389, 'x1']
y1 = data.at[389, 'y1']
ax.annotate('End 1', xy = (x1, y1), xytext = (x1, y1-0.1), arrowprops=dict(color='slategrey', shrink=0.01, width =0.01, headwidth = 4, headlength = 9), color = 'slategrey')
ax = data.plot(x="x2", y="y2", ax=ax)
x2 = data.at[389, 'x2']
y2 = data.at[389, 'y2']
ax.annotate('End 2', xy = (x2, y2), xytext = (x2, y2-0.05), arrowprops=dict(color='slategrey', shrink=0.01, width =0.01, headwidth = 4, headlength = 9), color = 'slategrey')
ax = data.plot(x="x3", y="y3", ax=ax)
# ax.axis([-0.75,0.2,-0.8,0.2])
x3 = data.at[389, 'x3']
y3 = data.at[389, 'y3']
ax.annotate('End 3', xy = (x3, y3), xytext = (x3+0.1, y3), arrowprops=dict(color='slategrey', shrink=0.01, width =0.01, headwidth = 4, headlength = 9), color = 'slategrey')
ax.set_facecolor('xkcd:charcoal')
plt.show()