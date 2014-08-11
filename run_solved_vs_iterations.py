#!/usr/bin/env python

from __future__ import print_function
from sys import argv as args
from sys import exit
import dm_tests as dmt

if __name__ == "__main__":
    try:
        path = args[1]
    except:
        print("Invalid command line arguments.")
        exit(1)

    dmt.solved_vs_iterations(path)
