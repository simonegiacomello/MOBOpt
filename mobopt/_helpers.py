# -*- coding: utf-8 -*-

import matplotlib.pyplot as pl
import numpy as np
try:
    # try importing the C version
    from deap.tools._hypervolume import hv
except ImportError:
    # fallback on python version
    from deap.tools._hypervolume import pyhv as hv

# % Clip x between xmin and xmax
def clip(x, xmin, xmax):
    for i, xx in enumerate(x):
        if xmin[i] is not None:
            if xx < xmin[i]:
                x[i] = xmin[i]
        if xmax[i] is not None:
            if xx > xmax[i]:
                x[i] = xmax[i]
    return


# % Visualiza
def plot_1dgp(fig, ax, space, iterations, Front, last):

    ax.clear()

    PF = space.f

    lineObs, = ax.plot(-PF[:, 0], -PF[:, 1], 'o', label=f"N = {iterations}",
                       alpha=0.2)
    lineNsga, = ax.plot(Front[:, 0], Front[:, 1], 'o', label="NSGA", alpha=0.5)
    Last, =  ax.plot(Front[last, 0], Front[last, 1], '*')
    lastObs, = ax.plot(-PF[-1, 0], -PF[-1, 1], '*')

    ax.grid()
    pl.xlabel('$f_1$')
    pl.ylabel('$f_2$')
    pl.legend(loc='upper right')

    fig.canvas.draw()
    fig.canvas.flush_events()

def nondominated_pts(pts, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(pts.shape[0])
    n_points = pts.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(pts):
        nondominated_point_mask = np.any(pts>pts[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        pts = pts[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def sms_hv(front, ref=None):
    """Return the hypervolume of a *front*. If the *ref* point is not
    given, the worst value for each objective +1 is used.

    :param front: The population (usually a list of undominated individuals)
                  on which to compute the hypervolume.
    :param ref: A point of the same dimensionality as the individuals in *front*.
    """
    # Must use wvalues * -1 since hypervolume use implicit minimization
    wobj = front * -1
    if ref is None:
        ref = np.max(wobj, axis=0) + 1
    return hv.hypervolume(wobj, ref)