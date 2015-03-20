#!/usr/bin/env python

from __future__ import print_function

import sys
import json

import itertools as it

eprint = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

class CLIError(Exception):
    pass

if __name__ == "__main__":
    mma_path = None
    py_path = None
    i = 1
    try:
        while i < len(sys.argv):
            arg = sys.argv[i]
            n = lambda: sys.argv[i+1]
            if arg == '--mathematica':
                mma_path = n()
                i += 1
            elif arg == '--python':
                py_path = n()
                i += 1
            else:
                raise CLIError("unrecognized command-line argument: %s" % arg)
            i += 1
    except IndexError:
        eprint("unexpected end of command-line arguments.")
        sys.exit(2)
    except CLIError as e:
        eprint(e)
        sys.exit(2)

    if mma_path is None or py_path is None:
        eprint("please give both --mathematica and --python data")
        sys.exit(2)

    with open(mma_path, 'r') as f:
        mma_data = json.load(f)

    with open(py_path, 'r') as f:
        py_data = json.load(f)

    del mma_data['schwefel']

    functions = mma_data.keys()

    figure_data = {}

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, f in enumerate(functions, 1):
        figure_data[f] = {}

        mma_avg = 0
        py_avg = 0
        n = 0
        for m, p in it.izip(mma_data[f], py_data[f]):
            n += 1
            mma_avg += m["fun"]
            py_avg += m["fun"]
        mma_avg /= float(n)
        py_avg /= float(n)

        figure_data[f]['mma'] = mma_avg
        figure_data[f]['py'] = py_avg

    ax.set_tick_labels(*functions)

    for f in functions:
        ax.bar(range(len(functions)), map(lambda f: ['mma'], figure_data[f])
