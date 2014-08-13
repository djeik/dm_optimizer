#!/usr/bin/env python

from sys import argv as args
import dm_tests as dmt

if __name__ == "__main__":
    if len(args) < 2:
        raise ValueError("Wrong number of arguments.")

    data_dir = args[1]
    dmt.solved_vs_iterations_plots(data_dir)
