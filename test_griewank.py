#!/usr/bin/env python

from __future__ import print_function

import test_functions as tf
import dm_optimizer_simon as dm

from sys import stdout, stdin

import json

def run(niter, startpoints):
    return dm.dm(tf.griewank, niter, dim=len(startpoints[0]), startpoints=startpoints)

if __name__ == "__main__":
    settings = json.load(stdin)
    json.dump(run(settings["niter"], settings["startpoints"], stdout, indent=4)
