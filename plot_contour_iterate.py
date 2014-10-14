#!/usr/bin/env python

from __future__ import print_function

import dm_tests        as dmt
import dm_tests_config as dmtc
import cPickle         as cp
import deap.benchmarks as bench
import multiprocessing as mp

from os        import path
from datetime  import datetime
from glob      import iglob
from itertools import repeat

from dm_utils import unwrap_bench

import jerrington_tools as jt

def make_experiment_dir():
    """ Create a directory `results/<datetime>` and return its path. """
    return jt.mkdir_p(path.join("results", datetime.now().isoformat()))

def iget_optimize_results(dir_path):
    """ Given a path to the log directory for a given test, this function
        lazily unpickles files of the form 'N-iterate.pickle'.
        """
    for p in iglob(path.join(dir_path, "*-result.pickle")):
        yield cp.load(p)

def is_successful(res, test_info, experiment_settings):
    if not hasattr(res, "fun"):
        return False

    return (res.fun - test_info["optimum"]) ** 2 <= \
            experiment_settings["success_threshold"]**2

def plot_iterate(plot_dir, plot_path, lpos, test):
    """ Plot the iterate positions from the given list on top of a contour
        plot of the objective function from the given test. """
    def nary2binary(f):
        """ Convert an n-ary function into a binary function, so that numpy
            vectorization can be applied properly. """
        return lambda x, y: f([x, y])

    print("Begin plotting:", test["name"], file=sys.stderr)
    f      = np.vectorize(nary2binary(eval(test["function"])))
    xs, ys = jt.splat(jt.supply(np.linspace, {"num":500}))(test["range"])
    X, Y   = np.meshgrid(xs, ys)
    Z      = f(X, Y)

    fig = plt.figure()
    fig.suptitle(test["name"])
    ax  = fig.subplot(1, 1, 1)
    ax.contourf(X, Y, Z)
    ax.plot(*zip(*map(jt.project_c(1), lpos)))
    fig.savefig(plot_path)

    print("End plotting:", test["name"], file=sys.stderr)

def main( (exp_dir_path, test) ):
    exp_dir = jt.mkdir_p(exp_dir_path)

    # we could use a constant function for True to plot everything
    plot_criterion = is_successful

    name, stats = dmt.experiment_task(
            (exp_dir, test, dmtc.optimizers["dm"], dmtc.poll_names) )
    plot_dir = jt.mkdir_p(path.join(exp_dir, name, "plots"))
    rs = iget_optimize_results(path.join(exp_dir, name, "log"))

    for i, r in enumerate(rs): # for each OptimizeResult object
        if plot_criterion(r, test, dmtc.experiment_defaults):
            plot_iterate(
                plot_dir, "".join([name, '-', str(i), ".pdf"]),
                    r.lpos, test)

if __name__ == "__main__":
    # we make a function that takes a dict, a key, and a value, and produces a
    # new dict in which that key has been set to the given value.
    def replace_value(d, k, v):
        d_ = dict(d)
        d_[k] = v
        return d_

    # We use this new function to set the `dimensions` entry for each test to
    # 2, and save the resulting list of dicts.
    my_tests = map(lambda d: replace_value(d, "dimensions", 2), dmtc.tests)

    exp_dir = make_experiment_dir()

    pool = mp.Pool(4)

    pool.map(main, zip(repeat(exp_dir), my_tests))
