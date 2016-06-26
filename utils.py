import subprocess
from tempfile import mkstemp

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz
from IPython.core.display import HTML


def plot_surface(clf, X, y,
                 xlim=(-10, 10), ylim=(-10, 10), n_steps=250,
                 subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
                         np.linspace(ylim[0], ylim[1], n_steps))

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    if show:
        plt.show()


def draw_tree(clf, feature_names, **kwargs):
    _, name = mkstemp(suffix='.dot')
    _, svg_name = mkstemp(suffix='.svg')
    export_graphviz(clf, out_file=name,
                    feature_names=feature_names,
                    **kwargs)
    command = ["dot", "-Tsvg", name, "-o", svg_name]
    subprocess.check_call(command)
    return HTML(open(svg_name).read())
