#!/usr/bin/env python2

from __future__ import print_function

import sys
from sys import argv as args
from sys import exit

import dm_tests as dmt

mkfprint = lambda f: lambda *x, **xs: print(*x, file=f, **xs)
errprint = mkfprint(sys.stderr)


def show_usage():
    map(errprint, ["run_test.py -- run the optimizer test suite.",
                  "usage: run_test.py <experiment directory> <optimizer tag>",
                  "This program is meant to be invoked through Reproducible in pipeline mode.",
                  "The experiment directory thus comes from Reproducible. See its documentation",
                  "for more details on how that is done.",
                  "Valid optimizer tags are:",
                  "\tdm -- Difference Map-based algorithm.",
                  "\tsa -- Simulated Annealing."])

if __name__ == "__main__":
    if len(args) < 3:
        errprint("fatal: incorrect command line.")
        show_usage()
        exit(1)

    print(args[1])

    #try:
        # run_test returns the string that is the path
    dmt.run_test(args[1], args[2])
    #except Exception as e:
    #    map(errprint, ["fatal: the test failed.",
    #                   "Inner error message:",
    #                   ("\t", e)])
    #    exit(1)
    #else:
    #    exit(0)
