#!/usr/bin/env python

from __future__ import print_function
from sys import argv as args
from sys import exit
from datetime import datetime
import dm_tests as dmt
from os import path

if __name__ == "__main__":
    try:
        out_dir = args[1]
    except:
        out_dir = str(datetime.now())

    out_dir = path.join("results/" + out_dir)

    dmt.solved_vs_iterations(out_dir)
