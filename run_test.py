#!/usr/bin/env python2

from __future__ import print_function

import sys
from sys import argv as args
from sys import exit

import dm_tests as dmt

mkfprint = lambda f: lambda *x, **xs: print(*x, file=f, **xs)
errprint = mkfprint(sys.stderr)

if __name__ == "__main__":
    if len(args) < 2:
        map(errprint, ["fatal: no test specified.",
                       "Please specify a test (dm or sa)."])
        exit(1)

    try:
        # run_test returns the string that is the path
        print(dmt.run_test(args[1]))
    except Exception as e:
        map(errprint, ["fatal: the test failed.",
                       "Inner error message:",
                       "\t" + str(e)])
        exit(1)
    else:
        exit(0)
