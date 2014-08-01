#!/usr/bin/env python2

import dm_tests as dmt
from sys import argv as args
from itertools import imap

match_any_of = lambda ys, x: any(imap(lambda y: x == y, ys))
mk_match_any = lambda x: lambda ys: match_any_of(ys, x)

if __name__ == "__main__":
    test_all = False
    show     = False
    dir      = args[1] # supplied by Reproducible

    i = 2
    while i < len(args):
        arg = args[i]
        nextarg = lambda: args[i+1]
        arg_match = mk_match_any(arg)
        if arg_match(["-a", "--test-all"]):
            test_all = True
        elif arg_match(["-i", "--show-interactive"]):
            show = True
        else:
            raise ValueError("fatal: unrecognized command line option ``%s''." % arg)

    dmt.dm_plot_3d(dir, test_all, show)
