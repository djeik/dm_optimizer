#!/usr/bin/env python

from __future__ import print_function

import dm_utils as dmu
unwrap_bench = dmu.unwrap_bench

import deap.benchmarks as bench

import dm_optimizer as dm
import dm_tests as dmt

from sys import argv as args
from sys import stderr, exit

if __name__ == "__main__":
    if len(args) < 3:
        p = lambda *args, **kwargs: print(*args, file=stderr, **kwargs)

        map(p, ["Insufficient arguments.",
            "usage: ./run_quick_test.py <test_name> <optimizer name>"])
        exit(1)
    r = dmt.run_single_test(dmt.get_test_by_name(dmt.tests, args[1]), args[2])
    print(r)
