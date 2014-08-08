from __future__ import print_function

from sys import argv as args
from os import path

import dm_utils as dmu

NAME_COL = 0
PERF_COL = 4

def main(args):
    dir = args[1]
    dm_avgs_path = path.join(dir, "..", "dm", "averages.txt")
    sa_avgs_path = path.join(dir, "..", "sa", "averages.txt")

    # This is assuming that there is the same number of entries in both averages files, and that the order of the functions is the same therein.
    parse = lambda f: map(lambda l: (l[NAME_COL], l[PERF_COL]), dmu.csv_to_tuples([line.strip("\n") for line in f.readlines()[1:]]))

    with open(dm_avgs_path) as dm_f:
        dm_perf = parse(dm_f)

    with open(sa_avgs_path) as sa_f:
        sa_perf = parse(sa_f)

    perf_ratios = ((dm_stat[0], dm_stat[1] / sa_stat[1]) if sa_stat[1] != 0 else 0 for (dm_stat, sa_stat) in zip(dm_perf, sa_perf))

    with open(path.join(dir, "comparison.txt"), 'w') as f:
        dmu.print_csv("test", "dm/sa performance", file=f)
        for (name, ratio) in perf_ratios:
            dmu.print_csv(name, ratio, file=f)

if __name__ == "__main__":
    main(args)
