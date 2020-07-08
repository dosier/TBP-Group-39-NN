import pickle

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy


def label(xy, dxy, text):
    x = xy[0] + dxy[0]
    y = xy[1] + dxy[1]  # shift y-value for label so that it's below the artist
    plt.text(x, y, text, ha="center", family='sans-serif', size=14, color='white')


def plot_model_performance(history_file_path):
    history = numpy.load(history_file_path, allow_pickle=True)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_initial_positions():
    ax = plt.gca()
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1, 1])
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, [0.0, 1.0]),
        (Path.LINETO, [-0.5, 0.8]),
        (Path.LINETO, [-0.5, 0.0]),
        (Path.LINETO, [0.0, 0.0]),
        (Path.LINETO, [0.0, 1.0])
    ]
    codes, vertices = zip(*path_data)
    path = mpath.Path(vertices, codes)
    patch = mpatches.PathPatch(path, color=u'#ff7f0e', alpha=0.2)
    ax.add_patch(patch)
    ax.add_patch(mpatches.FancyArrow(0.0, 0.0, 1.0 - 0.120, 0.0, width=0.005, head_width=0.05, color='grey'))
    ax.add_patch(mpatches.FancyArrow(0.0, 0.0, -0.25 + 0.095, 0.25 - 0.095, width=0.005, head_width=0.05, color='grey'))
    ax.add_patch(
        mpatches.FancyArrow(0.0, 0.0, -0.75 + .1125, -0.25 + .0375, width=0.005, head_width=0.05, color='grey'))
    # add an ellipse
    ax.add_patch(mpatches.Ellipse((1.0, 0.0), 0.075, 0.075, color=u'#1f77b4'))
    ax.add_patch(mpatches.Ellipse((-0.25, 0.25), 0.075, 0.075, color=u'#ff7f0e'))
    ax.add_patch(mpatches.Ellipse((-0.75, -0.25), 0.075, 0.075, color=u'#2ca02c'))
    ax.set_facecolor('xkcd:charcoal')
    label((1.0, 0.0), (0, -.15), "b1")
    label((-0.25, 0.25), (0, .10), "b2")
    label((-0.75, -0.25), (0, -.15), "b3")
    plt.show()


# plot_initial_positions()

plot_model_performance("tbp-mlp-10-layers-128-units-multi-in-2_history")
