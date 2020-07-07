import pandas
from matplotlib import pyplot as plt


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


data = pandas.read_csv("predictions/epoch-499.csv", index_col=None, header=0)
timesteps = drange(0, 3.9, 0.01, round_decimal_places=3)
data.insert(0, "timestep", timesteps)

ax = data.plot(x="x1", y="y1")
ax.annotate('End 1', (data.at[389, 'x1'], data.at[389, 'y1']))
ax = data.plot(x="x2", y="y2", ax=ax)
ax.annotate('End 2', (data.at[389, 'x2'], data.at[389, 'y2']))
ax = data.plot(x="x3", y="y3", ax=ax)
ax.annotate('End 3', (data.at[389, 'x3'], data.at[389, 'y3']))
plt.show()