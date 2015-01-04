#!/usr/bin/env python

""" Generate tons of test data as JSON. """

from __future__ import print_function

import dm_optimizer_simon as dms
import test_functions as tf
import jerrington_tools as jt
import sys
from os import path
from datetime import datetime
from time import time
from itertools import repeat
import json
import multiprocessing as mp
import numpy as np

RUNS_COUNT = 50

NITERS = [5, 10, 50, 100, 250, 500]
DISTS = [5, 10, 100, 500, 1000, 10000]
ALL_SOLVERS = ["dm", "sa"]

DEFAULT_RESULTS_DIR = "/home/tsani/projects/dm/results"

def do_test(kwargs):
    experiment_dir, solver, test = [kwargs[s] for s in ["experiment_dir", "solver", "test"]]

    def arg_default(arg_name, default_value):
        try:
            v = kwargs[arg_name]
        except KeyError:
            v = default_value
        return v

    niters = arg_default("niters", NITERS)
    dists = arg_default("dists", DISTS)
    options = arg_default("options", {})

    solver_function = getattr(dms, solver["name"])
    solver_dir = jt.mkdir_p(path.join(experiment_dir, solver["name"]))

    print("begin", test["name"], sep='\t', file=sys.stderr)

    test_dir = jt.mkdir_p(path.join(solver_dir, test["name"]))
    for niter in niters:
        niter_dir = jt.mkdir_p(path.join(test_dir, str(niter)))
        for dist in dists:
            dist_dir = jt.mkdir_p(path.join(niter_dir, str(dist)))

            if path.exists(path.join(dist_dir, "result")):
                continue

            fun = getattr(tf, test["function"])
            dim = test["dimensions"] or \
                    tf.SAMPLER_DEFAULTS["dimensions"]

            try:
                start_time = time()
                results = [solver_function(
                            fun,
                            niter,
                            dim=dim,
                            distance=dist,
                            **options)
                        for _ in xrange(RUNS_COUNT)]
                end_time = time()
                average_time = (end_time - start_time) / RUNS_COUNT
            except Exception as e:
                print("optimization failed:", solver["name"], test["name"], niter,
                        dist)
            else:
                summary = {
                        "statistics": {
                            "average_time": average_time
                        },
                        "settings": {
                            "solver": solver["name"],
                            "test": test["name"],
                            "niter": niter,
                            "distance": dist,
                            "run_count": RUNS_COUNT,
                        },
                        "run_data": results
                }
                with open(path.join(dist_dir, "result"), 'w') as f:
                    json.dump(summary, f)

    print("end", test["name"], sep='\t', file=sys.stderr)

def main(kwargs):
    experiment_dir = kwargs["experiment_dir"]

    for solver in kwargs["solvers"]:
        pool = mp.Pool(3)
        pool.map(do_test, map(
            lambda test: {
                "test": test,
                "experiment_dir": experiment_dir,
                "solver": solver
            }, kwargs["tests"]))


if __name__ == "__main__":
    experiment_dir = None
    experiment_dir_suffix = None
    solver_names = []
    tests = []
    options = {}

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        n = lambda: sys.argv[i+1]
        if arg == '-o':
            experiment_dir = n()
            i += 1
        elif arg == '--suffix':
            experiment_dir_suffix = n()
            i += 1
        elif arg == '--solvers':
            solver_names = n().split(',')
            i += 1
        elif arg == '--tests':
            tests = n().split(',')
            i += 1
        elif arg == '--options':
            options = json.loads(n())
            i += 1
        i += 1

    # default: in the default results dir (global constant), make a directory
    # named with the current date and time
    experiment_dir = experiment_dir or jt.mkdir_p(
            path.join(DEFAULT_RESULTS_DIR, str(datetime.now())))

    solver_names = solver_names or ALL_SOLVERS # default: assume all solvers
    for s in solver_names:
        if s not in ALL_SOLVERS:
            print("Solver", s, "not implemented.", file=sys.stderr)
            sys.exit(1)

    # Make a list of solver dicts, with their name and any options if given on
    # the command line
    solvers = []
    for solver_name in solver_names:
        try:
            solver_options = options[solver_name]
        except KeyError:
            solver_options = {}

        solvers.append({
            "name": solver_name,
            "options": solver_options
        })

    # Add the suffix if there is one.
    if experiment_dir_suffix is not None:
        experiment_dir += "-" + experiment_dir_suffix

    if tests: # if a list of test-functions was given on the command line, then
        # get the entries from the test-functions dictionary
        tests_ = []
        for t in tests:
            try:
                tests_.append(tf.tests_dict[t])
            except KeyError:
                print("Objective function", t, "is not implemented.", file=sys.stderr)
                sys.exit(1)
        tests = tests_
    else:
        tests = tf.tests_list # default: assume all tests

    main({
        "experiment_dir": experiment_dir,
        "options": options,
        "tests": tests,
        "solvers": solvers
    })
