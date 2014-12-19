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

DEFAULT_RESULTS_DIR = "/home/tsani/projects/dm/results"

def do_test(args):
    experiment_dir, solver_name, test = args
    solver = getattr(dms, solver_name)
    solver_dir = jt.mkdir_p(path.join(experiment_dir, solver_name))

    print("begin", test["name"], sep='\t', file=sys.stderr)

    test_dir = jt.mkdir_p(path.join(solver_dir, test["name"]))
    for niter in NITERS:
        niter_dir = jt.mkdir_p(path.join(test_dir, str(niter)))
        for dist in DISTS:
            dist_dir = jt.mkdir_p(path.join(niter_dir, str(dist)))

            if path.exists(path.join(dist_dir, "result")):
                continue

            fun = getattr(tf, test["function"])
            dim = test["dimensions"] or \
                    tf.SAMPLER_DEFAULTS["dimensions"]

            try:
                start_time = time()
                results = [solver(
                            fun,
                            niter,
                            dim=dim,
                            distance=dist)
                        for _ in xrange(RUNS_COUNT)]
                end_time = time()
                average_time = (end_time - start_time) / RUNS_COUNT
            except Exception as e:
                print("optimization failed:", solver_name, test["name"], niter,
                        dist)
            else:
                summary = {
                        "average_time": average_time,
                        "solver": solver_name,
                        "test": test["name"],
                        "niter": niter,
                        "distance": dist,
                        "run_count": RUNS_COUNT,
                        "run_data": results
                }
                with open(path.join(dist_dir, "result"), 'w') as f:
                    json.dump(summary, f)

    print("end", test["name"], sep='\t', file=sys.stderr)

def main(experiment_dir):
    for solver_name in ["dm", "bh"]:
        pool = mp.Pool(3)
        pool.map(do_test, zip(repeat(experiment_dir), repeat(solver_name),
            tf.tests_list))

if __name__ == "__main__":
    experiment_dir = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        n = lambda: sys.argv[i+1]
        if arg == '-o':
            experiment_dir = n()
            i += 1
        i += 1

    experiment_dir = experiment_dir or \
            jt.mkdir_p(path.join(
                DEFAULT_RESULTS_DIR, str(datetime.now())))

    main(experiment_dir)
