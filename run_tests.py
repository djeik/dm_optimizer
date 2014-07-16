#!/usr/bin/env python2

from __future__ import print_function

import dm_tests as dmt
import subprocess

from sys import argv as args
from sys import exit
import sys
from os import path

from itertools import islice

if __name__ == "__main__":
    force = False
    optimizer = None

    try:
        for (i, arg) in enumerate(islice(args, 1, None)):
            if any(map(lambda x: arg == x, ["-f", "--force"])):
                force = True
            else:
                optimizer = arg
        if args[1] == "-f" or args[1] == "--force":
            force = True
    except: # index exception only, i.e. the arg is not present.
        pass

    if not (path.exists("dm_optimizer.py") and path.exists("dm_tests.py")):
        print("fatal: optimizer or optimizer test suite does not exist.", file=sys.stderr)
        exit(1)

    if not optimizer:
        print("Please specify an optimizer code to use:")
        map(lambda k: print("\t", k, sep=''), dmt.optimizers)
        exit(1)

    if not optimizer in dmt.optimizers:
        print("The specified optimizer is invalid.")
        exit(1)

    git_status = subprocess.Popen(["git", "status", "--short", "dm_optimizer.py", "dm_tests.py"], stdout=subprocess.PIPE)
    out, err = git_status.communicate()
    git_status.wait()
    if git_status.returncode != 0:
        print("fatal: checking optimizer git repository status failed.", file=sys.stderr)
        exit(1)

    clean = len(out) == 0
    if (not clean) and (not force):
        print("The repository is not clean! Running this experiment does not guarantee reproducibility. Please commit your changes to dm_optimizer.py or dm_tests.py, or force the test with the -f switch.")
        exit(1)

    #try:
    edir = dmt.conduct_all_experiments(dmt.optimizers[optimizer]) # returns the experiment directory's path
    #except Exception as e:
    #    print("fatal: optimizer test suite failed:\n\t", e, sep='', file=sys.stderr)
    #    exit(1)

    git_rev_parse = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    out, err = git_rev_parse.communicate()
    git_rev_parse.wait()

    if git_rev_parse.returncode != 0:
        failure_message = "Unable to get SHA for this commit. Consider this experiment invalid."
        print(failure_message, file=sys.stderr)
        print("This message will be written to the experiment folder.", file=sys.stderr)
        with open(edir + "/sha-error.txt", 'w') as f:
            print(failure_message, file=f)

    with open(edir + "/rev.txt", 'w') as f:
        print(out, file=f)
        if not clean:
            print("NOT CLEAN", file=f)

