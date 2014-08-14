#!/usr/bin/env python

from __future__ import print_function
from sys import argv as args
from sys import exit
from datetime import datetime
import dm_tests as dmt
from os import path

from itertools import imap

eq = lambda x: lambda y: x == y
mk_e = lambda x: lambda s: any(imap(eq(x), s)) if hasattr(s, "__iter__") else s == args[i]

def main(args):
    out_dir = path.join("results", str(datetime.now()))
    step_scale = 0.5

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
        else:
            print("Unrecognized argument:", args[i])
        i += 1

    dmt.solved_vs_iterations(out_dir, extra_settings={"dm":{"stepscale_constant":step_scale}})

if __name__ == "__main__":
    main(args)
