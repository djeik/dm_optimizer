#!/usr/bin/env python

""" Don't run this script !
    It's a one-shot thing to regenerate SA output for success-vs-iterations
    for the plots for the paper. Turns out that SA was run with 500 iterations, but
    dm with only 250, so SA was unfairly outperforming us each time.
    """

import dm_tests_config as dmtc
import os
from os import path
import subprocess as sp
from sys import argv as args

if __name__ == "__main__":
    t = args[1]
    for test in dmtc.tests:
        sp.call(["Qmsub", "-n", str(dmtc.mp_subproc_count),
                          "-h", "16",
                          "./run_solved_vs_iterations_inner.py",
                          "-o",       "\"" + t + "/sa\"",
                          "-t",       test["name"],
                          "--solver", "sa"])
