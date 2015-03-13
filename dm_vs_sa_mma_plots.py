#!/usr/bin/env

import matplotlib.pyplot as plt
import numpy as np

import json
import sys

from collections import defaultdict

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    solvers = data.keys()
    functions = data['dm'].keys()

    results = defaultdict(lambda: defaultdict(dict))

    for solver_name, solver_data in data.items():
        for function_name, function_runs in solver_data.items():
            average_solution = sum(r['fun'] for r in function_runs) / \
                    float(len(function_runs))
            average_nfev = sum(r['nfev'] for r in function_runs) / \
                    float(len(function_runs))
            min_solution = min(r['fun'] for r in function_runs)
            results[solver_name][function_name]['average_solution'] = \
                    average_solution
            results[solver_name][function_name]['average_nfev'] = \
                    average_nfev
            results[solver_name][function_name]['min_solution'] = min_solution

    print json.dumps(results, indent=2)

    X = np.arange(len(functions))
    w = 0.9 / len(functions)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, (solver_name, solver_data) in enumerate(data.items()):
        ax.bar(X + i*w,
                [results[solver_name][s]['average_solution']
                    for s in functions],
                w)

    fig.savefig('out.pdf')
