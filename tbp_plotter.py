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


data = pandas.read_csv("predictions/output_182_normalized.csv", index_col=None, header=0)

x1 = data.at[389, 'x1']
y1 = data.at[389, 'y1']
x2 = data.at[389, 'x2']
y2 = data.at[389, 'y2']
x3 = data.at[389, 'x3']
y3 = data.at[389, 'y3']

# plot x and y coordinates of the first body
ax = data.plot(x="x1", y="y1")
ax.annotate('End 1', xy=(x1, y1), xytext=(x1, y1 - 0.1),
            arrowprops=dict(color='slategrey', shrink=0.01, width=0.01, headwidth=4, headlength=9), color='slategrey')

# plot x and y coordinates of the second body
ax = data.plot(x="x2", y="y2", ax=ax)
ax.annotate('End 2', xy=(x2, y2), xytext=(x2, y2 - 0.05),
            arrowprops=dict(color='slategrey', shrink=0.01, width=0.01, headwidth=4, headlength=9), color='slategrey')

# plot x and y coordinates of the third body
ax = data.plot(x="x3", y="y3", ax=ax)
ax.annotate('End 3', xy=(x3, y3), xytext=(x3 + 0.1, y3),
            arrowprops=dict(color='slategrey', shrink=0.01, width=0.01, headwidth=4, headlength=9), color='slategrey')

ax.set_facecolor('xkcd:charcoal')

plt.show()
