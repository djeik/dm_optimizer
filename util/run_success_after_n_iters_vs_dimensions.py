#!/usr/bin/env python

from __future__ import print_function

from sys import argv as args

import dm_tests         as dmt
import dm_tests_config  as dmtc

import jerrington_tools as j

from operator import eq
import os
from os import path

RESULTS_FORMAT = ("DIMENSIONS", "FAILURES", "SUCCESS", "TIME", "EVALS")

def print_results_to(results, handle):
    csv_print = lambda *args, **kwargs: print(
            *args, sep=',', file=handle, **kwargs)

    csv_print(*RESULTS_FORMAT)

    for stats in stass:
        csv_print(*stats[:5])

def run(solver_dir, solver_name, runs, test_name):
    """ The iteration count is set via the configuration file
        dm_tests_config.py.
        """
    test = filter(
            # Project the name and compare to the given one
            j.compose(j.curry2(eq)(test_name), j.dproject_c("name")),
            dmtc.tests)

    if len(test) != 1:
        raise ValueError("invalid name ``%s''" % test_name)

    test = test[0]

    test_dir_path = j.mkdir_p(path.join(solver_dir, test["name"]))

    experiment_settings = dict(dmtc.experiment_defaults)
    experiment_settings["runs"] = runs

    statss = []

    for dim in xrange(30):
        test["dimensions"] = dim
        statss.append(
                ((dim,) + dmt.conduct_experiment(
                    test_dir_path, test, dmtc.optimizers[solver_name],
                    experiment_settings)))

    j.with_file(
            j.curry2(print_results_to)(statss),
            path.join(test_dir_path, test["name"] + ".csv"))

def main(args):
    solver_dir  = None
    solver_name = None
    runs        = None
    test_name   = None

    i = 1
    while i < len(args):
        arg = args[i]
        nextarg = lambda: args[i + 1]
        aeq = lambda x: (
                any(map(j.curry2(eq)(arg), x)) if hasattr(x, "__iter__")
                else x == arg)
        if aeq(["--output-dir", "-o"]):
            solver_dir = nextarg()
            i += 1
        elif aeq(["--solver", "-s"]):
            solver_name = nextarg()
            i += 1
        elif aeq(["--runs", "-r"]):
            runs = int(nextarg())
            i += 1
        elif aeq(["--test", "-t"]):
            test_name = nextarg()
            i += 1
        else:
            j.errprint(
                    "Unrecognized command line argument ``", arg, "''.",
                    sep='')
        i += 1

    if any(map(
            lambda x: x is None,
            [solver_dir, solver_name, runs, test_name])):
        j.errprint("Insufficient command line arguments given.")
        return

    run(solver_dir, solver_name, runs, test_name)


if __name__ == "__main__":
    main(args)
