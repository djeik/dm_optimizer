#!/usr/bin/env python

from __future__ import print_function
from sys import argv as args
from sys import exit, stderr
from datetime import datetime
from os import path

from itertools import imap

import dm_tests        as dmt
import dm_tests_config as dmtc
import dm_utils        as dmu

""" Curryable version of a function defined by
        lambda x, y: x == y
    """
eq = lambda x: lambda y: x == y

""" From a given value, construct a function that will accept either a single value
    or an iterable: in the former case, the function will check whether the two
    values are equal, and in the latter case, it will check the equality of the first
    value against all the values in the iterable. Again in the latter case, a summary
    truth value will be computed from the ``checkf'' parameter, which defaults to
    ``any''.
    """
mk_e = lambda x, checkf=any: lambda s: (any(imap(eq(x), s))
                                        if hasattr(s, "__iter__") else
                                        s == args[i])

if __name__ == "__main__":
    out_dir = path.join("results", str(datetime.now()))
    step_scale = 0.5

    solver = None
    test_name = None

    i = 1
    while i < len(args):
        e = mk_e(args[i]) # construct a comparison function ``e'' for this arg.
        n = lambda: args[i+1] # use a lambda to avoid index errors at end of args

        if e(["-o", "--output-dir"]):
            out_dir = n()
            i += 1
        elif e(["-s", "--step-scale"]):
            step_scale = float(n())
            i += 1
        elif e(["--solver"]):
            solver = n()
            i += 1
        elif e(["-t", "--test"]):
            test_name = n()
            i += 1
        else:
            print("Unrecognized argument:", args[i])
            exit(1)
        i += 1

    if solver is None or test_name is None:
        print("Missing solver name or test name.", file=stderr)

    extra_settings = {"dm":{"stepscale_constant":step_scale}}
    extra_settings = extra_settings[solver] if solver in extra_settings else {}
    solver_dir = path.join(out_dir, solver)
    test = filter(lambda t: t["name"] == test_name, dmtc.tests)[0]

    alives_vs_t = dmt.solved_vs_iterations_inner(solver_dir, solver, test, extra_settings)

    solver_results_dir = path.join(solver_dir, "results")
    dmu.mkdir_p(solver_results_dir)
    test_result_path = path.join(solver_results_dir, test_name + ".csv")

    dmu.with_file(
            lambda f: [dmu.print_csv(count, file=f) for count in alives_vs_t],
            test_result_path, 'w')

    dmu.with_file(
            lambda f: print(args[1:], file=f),
            path.join(solver_results_dir, "invocation-" + test_name + ".txt"), 'w')
