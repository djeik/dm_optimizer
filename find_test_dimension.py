#!/usr/bin/env python2

from __future__ import print_function
import dm_tests_config as dmtc
import dm_tests as dmt
import util.dm_utils as dmu

from util.dm_utils import unwrap_bench
import deap.benchmarks as bench

from time import time

import multiprocessing as mp

run_count = 50
success_cutoff = 0.2

def find_best_dimension_for(test, options=None):
    if test["dimensions"] != None:
        raise ValueError("The objective fuction must not have a predetermined number of dimensions to use.")
    sa = dmtc.optimizers["sa"]
    sa_opt = sa["optimizer"]
    sa_conf = options or sa["config"]
    success_threshold = dmtc.experiment_defaults["success_threshold"]
    range = test["range"] or dmtc.sampler_defaults["range"]
    for d in xrange(1, 100):
        if d % 5 == 0:
            dmu.errprint(test["name"] + ":", d)
        rs = []
        start_time = time()
        for _ in xrange(run_count):
            rs.append(sa_opt(eval(test["function"]), d, range, sa_conf))
        end_time   = time()
        (failures, success_rate, time_avg, nfev_avg) = dmt.calculate_stats(test, rs, end_time - start_time)
        if success_rate < success_cutoff:
            return d
    return -1

def finder_wrapper(test):
    d = find_best_dimension_for(test, {"niter":100})
    print(test["name"], d, sep =',')
    return (test["name"], d)

def find_best_dimensions():
    pool = mp.Pool()
    ds = dict(pool.map(finder_wrapper, filter(lambda t: t["dimensions"] == None, dmtc.tests)))
    print(ds)
    return ds
