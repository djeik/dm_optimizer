#!/usr/bin/env python

from __future__ import print_function

from sys import argv as args

import os
from os import path
import test_functions
import matplotlib.pyplot as plt
from itertools import imap

import jerrington_tools as jt
from jerrington_tools import project

from dm_tests import parse_solved_vs_iterations_data_for_one_optimizer
from dm_tests_config import iterations_config, experiment_defaults


def solved_vs_iterations_plots_pro(path_to_data,
        iterations_count=iterations_config["end"],
        runs_count=experiment_defaults["runs"]):
    # we have data per stepsize stored in all-dm, in folders named with the
    # stepsize.
    path_to_stepsizes = path.join(path_to_data, "dm")

    has_dm = path.exists(path_to_stepsizes)

    if has_dm:
        stepsizes = map(
                lambda s: (float(s),
                           path.join(path_to_stepsizes, s)),
                os.listdir(path_to_stepsizes))

        maxsize = max(imap(jt.fst, stepsizes))

        # make the directory where we'll save the plots
        plot_dir = path.join(path_to_data, "plots")
        jt.mkdir_p(plot_dir)

        # dict like D[stepsize][function_name] : list representing fraction
        # completed by time
        functions_by_stepsize = map(
            lambda (stepsize, directory): (
                stepsize,
                parse_solved_vs_iterations_data_for_one_optimizer(
                    directory, runs_count)),
            stepsizes)

    # now we need to do the same but for SA
    simulated_annealing_path = path.join(
            path_to_data, "sa")

    has_sa = path.exists(simulated_annealing_path)

    if has_sa: # if there is SA data, then we parse it
        sa_data = parse_solved_vs_iterations_data_for_one_optimizer(
                simulated_annealing_path, runs_count)

    if not (has_sa or has_dm): # If there is no data, then there's nothing to plot
        raise ValueError("No data to plot in %s" % path_to_data)

    for test in test_functions.tests_list:
        if has_sa:
            if not (test["name"] in sa_data and all(
                    imap(
                        lambda d: test["name"] in d,
                        imap(
                            project(1),
                            functions_by_stepsize)))):
                print("Skipping function", test["name"],
                        "because one or more datasets don't have entries for it.")
                continue
        fig = plt.figure()
        fig.suptitle("%s (%d dimensions) - "
                "fraction of successful runs vs. time"
                % (test["name"], test["dimensions"]))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0, iterations_count)
        ax.set_ylim(0, 1)

        if has_sa:
            ax.plot(sa_data[test["name"]], label="sa", color=(0, 0, 1))

        if has_dm:
            for (size, dm_data) in functions_by_stepsize:
                jt.errprint("Plotting for size", size)
                ax.plot(dm_data[test["name"]], label="dm %f" % size,
                        color=(size/maxsize, 0.5, 0.5))
        fig.savefig(path.join(plot_dir, test["name"] + ".eps"))
        plt.close(fig) # close the figure to save memory

if __name__ == "__main__":
    solved_vs_iterations_plots_pro(args[1])
