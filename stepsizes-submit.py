#!/usr/bin/env python

import subprocess as sp
import numpy as np

from sys import stdout, stderr

if __name__ == "__main__":
    for stepscale in np.arange(0.1, 1.31, 0.1):
        sp.call(["Qmsub", "./run_solved_vs_iterations.py", "-s", str(stepscale)], stdout=stdout, stderr=stderr)
