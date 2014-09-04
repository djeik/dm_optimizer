#!/usr/bin/env python

from sys import argv as args
from datetime import datetime
from os import path

import subprocess as sp

import dm_tests_config as dmtc
import jerrington_tools as j

VARIABLE_DIMENSION_TESTS = [
        "ackley",
        "bohachevsky",
        "cigar",
        "sphere",
        "griewank",
        "rastrigin",
        "rosenbrock",
        "schaffer",
        "schwefel"]

def first(pred, seq):
    for elm in seq:
        if pred(elm):
            return elm
    raise ValueError("no elements in the given sequence satisfy the predicate")

def main(args):
    good_tests = j.for_each(VARIABLE_DIMENSION_TESTS,
            lambda t: first(lambda t_: t_["name"] == t, dmtc.tests))

    experiment_dir = j.mkdir_p(
        path.join(
            "results", str(datetime.now()).replace(" ", "-") + "-pvsd"))

    for solver in ["dm", "sa"]:
        solver_dir = j.mkdir_p(path.join(experiment_dir, solver))

        for test in good_tests:
            sp.call(["Qmsub", "-n", "1", "-h", "48",
                     "./run_success_after_n_iters_vs_dimensions.py",
                     "--solver", solver,
                     "--output-dir", solver_dir,
                     "--test", test["name"],
                     "--runs", "250"])

if __name__ == "__main__":
    main(args)
