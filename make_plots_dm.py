#!/usr/bin/env python2

from __future__ import print_function

import sys
from sys import argv as args
from sys import exit

import dm_tests as dmt
import jerrington_tools as j

if __name__ == "__main__":
    if len(args) < 2:
        j.errprint("fatal: incorrect command line")
        exit(1)

    dmt.generate_all_dm_plots(args[1])
