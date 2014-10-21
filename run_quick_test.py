#!/usr/bin/env python

from __future__ import print_function

"""
In [1]: import dm_utils as dmu

In [2]: unwrap_bench = dmu.unwrap_bench

In [3]: import deap.benchmarks as bench

In [4]: import dm_optimizer as dm

In [5]: import dm_tests as dmt
/lb/project/gravel/bin/anaconda/lib/python2.7/site-packages/matplotlib/__init__.py:464: UserWarning: matplotlibrc text.usetex can not be used with *Agg backend unless dvipng-1.5 or later is installed on your system
  warnings.warn( 'matplotlibrc text.usetex can not be used with *Agg '

          In [6]: dmt.run_test(dmt.get_test_by_name(dmt.tests, "ackley"), "dm")
"""

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
    r = dmt.run_test(dmt.get_test_by_name(dmt.tests, args[1]), args[2])
    print(r)
