#!/usr/bin/env python

from __future__ import print_function

import dm_utils as dmu
unwrap_bench = dmu.unwrap_bench

import deap.benchmarks as bench

import dm_optimizer as dm
import dm_tests as dmt

from sys import argv as args
from sys import stderr, exit

import test_functions

if __name__ == "__main__":
    if len(args) < 3:
        p = lambda *args, **kwargs: print(*args, file=stderr, **kwargs)

        map(p, ["Insufficient arguments.",
            "usage: ./run_quick_test.py <test_name> <optimizer name>"])
        exit(1)
    r = dmt.run_single_test(test_functions.tests_dict[args[1]], args[2])
    print(r)
