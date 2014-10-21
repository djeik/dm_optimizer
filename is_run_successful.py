#!/usr/bin/env python

""" A little script to determine from an OptimizeResult pickle whether that optimization run
    was successful by checking that the final function value from that run is less than
    the default success_threshold in difference with the desired optimum for that objective function.
"""

from __future__ import print_function

from sys import argv as args, exit

import jerrington_tools as jt
import cPickle          as cp
import dm_tests_config  as dmtc

if __name__ == "__main__":
    test_name = args[1]
    test = dmtc.get_test_by_name(dmtc.tests, test_name)

    def is_successful(res):
        return (res.fun - test["optimum"])**2 <= dmtc.experiment_defaults["success_threshold"]

    if jt.with_file(jt.compose(is_successful, cp.load), args[2]):
        exit(0)
    else:
        exit(1)

