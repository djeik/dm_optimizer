#!/usr/bin/env python2

import dm_tests as dmt
import subprocess

from sys import argv as args
from sys import exit

if __name__ == "__main__":
    force = False

    try:
        if args[1] == "-f" or args[1] == "--force":
            force = True
    except: # index exception only, i.e. the arg is not present.
        pass

    git_status = subprocess.Popen(["git", "status", "--short", "dm_optimizer.py"], stdout=subprocess.PIPE)
    out, err = git_status.communicate()
    git_status.wait()

    if (not force) and len(out) > 0:
        print("The repository is not clean! Running this experiment does not guarantee reproducibility. Please commit your changes to dm_optimizer.py or dm_tests.py, or force the test with the -f switch.")
        exit(1)

    dmt.conduct_all_experiments()
