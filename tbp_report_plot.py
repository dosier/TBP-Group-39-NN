import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt


def label(xy, text):
    y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)


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
patch = mpatches.PathPatch(path, color='grey')
ax.add_patch(patch)

ax.add_patch(mpatches.FancyArrow(0.0, 0.0, 1.0 - 0.120, 0.0, width=0.005, head_width=0.05, color='black'))
ax.add_patch(mpatches.FancyArrow(0.0, 0.0, -0.25 + 0.095, 0.25 - 0.095, width=0.005, head_width=0.05, color='black'))
ax.add_patch(mpatches.FancyArrow(0.0, 0.0, -0.75 + .1125, -0.25 + .0375, width=0.005, head_width=0.05, color='black'))

# add an ellipse
ax.add_patch(mpatches.Ellipse((1.0, 0.0), 0.075, 0.075, color='blue'))
ax.add_patch(mpatches.Ellipse((-0.25, 0.25), 0.075, 0.075, color='orange'))
ax.add_patch(mpatches.Ellipse((-0.75, -0.25), 0.075, 0.075, color='green'))

label((1.0, 0.0), "b1")
label((-0.25, 0.25), "b2")
label((-0.75, -0.25), "b3")

plt.show()
