#!/usr/bin/env

""" Python version of DMData.m, for collecting some data on a few objective
    functions and serializing to JSON.
"""

from __future__ import print_function

# We've already collected the Mathematica data, and to perform the fairest
# comparison possible, we'll use the exact same starting points by parsing
# the Mathematica data and extracting startpoints from there.

import numpy as np
import itertools as it
import test_functions as tf

import json
import sys
import dm_optimizer_simon as dm

dim = 4
niter = 100
tolerance = 1e-8
run_count = 50
range = [-250.0, 250.0]
range_size = range[1] - range[0]

eprint = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

def R():
    """ Get a random vector of `dim` dimensions with each x sampled
        uniformly from `range`.
    """
    return range_size * np.random.rand(dim) - (range_size/2.0)

def _list_repeat(fun, count):
    return map(lambda fun: fun(), (fun for _ in xrange(count)))

def test(function_name, start_points=None):
    """ Perform the test configured by the global variables on the given
        function, identified by name, optionally with the given starting
        points.
        If starting points are not provided, then they will be
        randomly sampled from the globally-configured range.
    """
    if len(start_points) != run_count:
        eprint("inconsistency: the configured number of runs does not match "
                "the number of start points provided.")
        sys.exit(1)

    # get the test function record from the test_functions module
    fun = tf.tests_dict[function_name]

    starts = start_points or _list_repeat(lambda: _list_repeat(R, 2), run_count)

    results = []
    for s in starts:
        r = dm.dm(getattr(tf, function_name), niter, tolerance, dim, startpoints=s)
        results.append(r)
    return results

class UnknownCLIArgumentError(Exception):
    def __init__(self, arg):
        self.arg = arg

    def __str__(self):
        return "unknown command line argument ``%s''" % self.arg

if __name__ == "__main__":
    mathematica_data_path = None
    i = 1
    try:
        while i < len(sys.argv):
            arg = sys.argv[i]
            n = lambda: sys.argv[i+1]
            if arg == "--source-data":
                mathematica_data_path = n()
                i += 1
            else:
                raise UnknownCLIArgumentError(arg)
            i += 1
    except UnknownCLIArgumentError as e:
        eprint(e)
        sys.exit(1)
    except IndexError:
        eprint("unexpected end of command line arguments")
        sys.exit(1)

    if mathematica_data_path is None:
        eprint("no mathematica data specified. Use --source-data.")
        sys.exit(1)

    mathematica_data = None

    try:
        with open(mathematica_data_path, 'r') as mathematica_data_file:
            mathematica_data = json.load(mathematica_data_file)
    except (OSError, IOError, ValueError) as e:
        eprint("failed to load source data:", e)
        sys.exit(1)

    python_results = {}
    for function_name, collected_data in mathematica_data.items():
        eprint(function_name)
        # Why minima[1] ? minima[0] is the position we get after minimizing starting
        # at minima[0]. Why minima[x][1]? Well, that's the x-value. The y-value is
        # stored in minima[x][0].
        starts = map(lambda r: (r["minima"][1][1], r["iterate"][0]), collected_data)
        python_results[function_name] = test(function_name, starts)

    json.dump(python_results, sys.stdout, indent=1)
