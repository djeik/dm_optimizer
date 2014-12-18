#!/usr/bin/env python

from __future__ import print_function

import sys

import dm_tests        as dmt
import dm_utils        as dmu
import dm_tests_config as dmtc
import cPickle         as cp
import multiprocessing as mp
import numpy           as np

from os        import path
from datetime  import datetime
from glob      import iglob
from itertools import repeat

from matplotlib import pyplot as plt

from sys import argv as args

import jerrington_tools as jt

import test_functions

def iget_optimize_results(dir_path):
    """ Given a path to the log directory for a given test, this function
        lazily unpickles files of the form 'N-iterate.pickle'.
        """
    for p in iglob(path.join(dir_path, "*-result.pickle")):
        yield jt.with_file(cp.load, p)

def is_successful(res, test_info, experiment_settings):
    """ A possible plotting criterion: whether the given optimization run found
        the global optimum.
        """
    if not hasattr(res, "fun"):
        return False

    return (res.fun - test_info["optimum"]) ** 2 <= \
            experiment_settings["success_threshold"]**2

def plot_iterate(ax, lpos, valsi, test):
    """ Plot the iterate positions from the given list on top of a contour
        plot of the objective function from the given test. """
    iterate_x, iterate_y = zip(*map(jt.snd, lpos))
    minimum_x, minimum_y = zip(*map(jt.snd, valsi))

    iterate_color = np.linspace(0, 1, len(iterate_x))
    minimum_color = np.linspace(0, 1, len(minimum_x))


    for i in xrange(len(iterate_x) - 1):
        ax.plot(
            iterate_x[i:i+2],
            iterate_y[i:i+2],
            color=(0, 0.5, iterate_color[i]))

    for i in xrange(len(minimum_x) - 1):
        ax.plot(
            minimum_x[i:i+2],
            minimum_y[i:i+2],
            color=(minimum_color[i], 0, 0.5))

def main( (exp_dir_path, test) ):
    exp_dir = jt.mkdir_p(exp_dir_path)

    # we could use a constant function for True to plot everything
    # or we can use is_successful to plot only runs that found the optimum
    plot_criterion = jt.const(True)

    name, stats = dmt.experiment_task(
            (exp_dir, test, dmtc.optimizers["dm"], dmtc.poll_names) )
    plot_dir = jt.mkdir_p(path.join(exp_dir, name, "plots"))
    rs = iget_optimize_results(path.join(exp_dir, name, "logs"))

    print(test["name"], ": begin computing contours.", sep='')
    # Create the points to plot for the function contours
    f      = np.vectorize(
            dmu.nary2binary(getattr(test_functions, test["function"])))
    xs1, xs2 = map(
            lambda r: jt.splat(jt.supply(
                np.linspace,
                {"num":jt.compose(dmtc.get_sample_count, r)(test)}))(
                test["range"] or list(repeat(
                    test_functions.SAMPLER_DEFAULTS["range"],
                    test["dimensions"] or test_function.SAMPLER_DEFAULTS["dimensions"]))),
            dmtc.get_range_sizes(test))
    X, Y   = np.meshgrid(xs1, ys1)
    Z      = f(X, Y)
    print(test["name"], ": end computing contours.", sep='')

    # Create the fiture
    fig = plt.figure(figsize=(10, 10), dpi=300)
    fig.suptitle(test["name"])
    ax  = fig.add_subplot(1, 1, 1)

    print(test["name"], ": begin plotting contours.", sep='')
    # Plot the function contours
    ax.contourf(X, Y, Z)
    print(test["name"], ": end plotting contours.", sep='')

    print(test["name"], ": begin plotting iterate and minima.", sep='')
    for i, r in enumerate(rs): # for each OptimizeResult object
        if plot_criterion(r, test, dmtc.experiment_defaults):
            # Plot the iterate and minima, save the figure, and remove the lines plotted

            plot_iterate(ax, r.lpos, r.valsi, test)
            plot_name = "".join([name, '-', str(i), ".pdf"])
            fig.savefig(path.join(plot_dir, plot_name))

            # Remove all the lines we've put onto the plot
            while ax.lines:
                ax.lines.pop(0)

    print(test["name"], ": end plotting iterate and minima.", sep='')

if __name__ == "__main__":
    # Get all the functions defined for 2 dimensions.
    my_tests = filter(
            lambda t: t["dimensions"] == 2,
            test_functions.tests_list)

    if len(args) == 2:
        exp_dir = args[1]
    else:
        exp_dir = dmu.make_experiment_dir("contour-plots")

    pool = mp.Pool(12)

    map(main, zip(repeat(exp_dir), my_tests))
