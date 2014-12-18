#!/usr/bin/env python

from __future__ import print_function
from sys import argv as args
from sys import exit, stderr
from datetime import datetime
import dm_tests as dmt
from os import path

from itertools import imap

eq = lambda x: lambda y: x == y
mk_e = lambda x: lambda s: any(imap(eq(x), s)) if hasattr(s, "__iter__") else s == args[i]

def show_usage():
    map(lambda m: print(m, file=stderr), [
        "run_solved_vs_iterations.py -- run the solved verses iterations data collector for every test.",
        "Provide a --solver, a step-scale for DM -s, and an output directory -o."])

def show_usage_die():
    show_usage()
    exit(1)

def main(args):
    out_dir = path.join("results", str(datetime.now()))
    step_scale = 0.5

    solver = None

    i = 1
    while i < len(args):
        e = mk_e(args[i])
        n = lambda: args[i+1]

        if e(["-o", "--output-dir"]):
            out_dir = n()
            i += 1
        elif e(["-s", "--step-scale"]):
            step_scale = float(n())
            i += 1
        elif e(["--solver"]):
            solver = n()
            i += 1
        else:
            print("Unrecognized argument:", args[i])
            return
        i += 1

    if solver is None:
        show_usage_die()

    dmt.solved_vs_iterations(out_dir, solver, extra_optimizer_config={"stepscale_constant":step_scale})

if __name__ == "__main__":
    main(args)
