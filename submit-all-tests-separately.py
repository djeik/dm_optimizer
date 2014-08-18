#!/usr/bin/env python

import subprocess as sp
import dm_tests_config as dmtc
from datetime import datetime

from numpy import arange

import os
from os import path

if __name__ == "__main__":
    for optimizer in dmtc.optimizers.keys():
        for test in dmtc.tests:
            for sscale in arange(0.1, 1.21, 0.1):
                sp.call(["Qmsub", "-n", str(dmtc.solved_vs_iterations_subproc_count),
                                  "-h", "16",
                                  "./run_solved_vs_iterations_inner.py",
                                  "-o", "results/" + "all-dm/" + str(sscale),
                                  "-s", str(sscale),
                                  "-t", test["name"],
                                  "--solver", optimizer])
