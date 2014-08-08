from __future__ import print_function

import matplotlib
matplotlib.use("TkAgg")

from datetime import datetime
import sys
import random

import os
from os import path

from time import time
from math import sin
from copy import copy

import numpy as np
from numpy.linalg import norm
import deap.benchmarks as bench # various test functions
from scipy.optimize import basinhopping

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from itertools import repeat, imap, ifilter, islice, chain, izip, takewhile

import dm_optimizer as dm
from dm_optimizer import dm_optimizer

from dm_tests_config import *
from dm_utils        import *

import multiprocessing as mp

sys.stdout = os.fdopen(sys.__stdout__.fileno(), 'w', 1) # line buffered output

# These are the functions Simon defined to test it on:
def simon_f1(xy):
    x, y = xy
    return 0.2 * (x**2 + y**2) / 2 + 10 * sin(x+y)**2 + 10 * sin(100 * (x - y))**2

def simon_f2(xs):
    xy = xs - np.array([100, 100])
    return simon_f1(xy)

# Visual debug tool for 3d
def plotf_3d(f, xyzs_, start=np.array([-1,-1]), end=np.array([1,1]), smoothness=1.0, autobound=True, autosmooth=True):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    z, xys = zip(*xyzs_) # prepare to draw the line
    x, y = zip(*xys)
    N = len(x)
    for i in xrange(N-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.jet(255*i/N))

    if autobound:
        start = reduce(np.minimum, xys)
        end = reduce (np.maximum, xys)

    if autosmooth:
        smoothness = norm(end - start) / 100.0

    print(start, end, smoothness)

    xs = np.arange(start[0], end[0], smoothness)
    ys = np.arange(start[1] + y[-1], end[1] + y[-1], smoothness)
    X, Y = np.meshgrid(xs, ys)
    zs = np.array([f((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, # draw the surface
            linewidth=0, antialiased=True)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig

def multiplot(dats, names=[], nrows=None, ncols=1):
    """ Make multiple plots in the case where each x value has several y values associated with it.
        If nrows is None, then the number of rows is calculated based on the size of the tuples in dats and
        the number of columns specified. """

    if not dats:
        raise ValueError("No data.")

    if len(dats[0]) == 1:
        raise ValueError("Can't plot a 1D object.")

    if len(names) != 0 and len(names) != len(dats[0]) - 1:
        raise ValueError("Incorrect number of names given for subplots.")

    if nrows is None:
        nrows = int(np.ceil((len(dats[0]) - 1) / float(ncols)))

    plt.subplots(nrows=nrows, ncols=ncols)
    plt.tight_layout()

    for i in xrange(1, len(dats[0])):
        plt.subplot(nrows, ncols, i)
        if names:
            plt.title(names[i-1])
        plt.plot(*zip(*map(lambda dat: (dat[0], dat[i]), dats)))

def calculate_stats(test, rs, time_total):
    optimization_type = test["optimization_type"]
    time_avg      = time_total / float(len(rs))
    nfev_total    = sum(imap(lambda r: r.nfev, rs))
    nfev_avg      = nfev_total / float(len(rs))
    success_total = len(filter(lambda r: r.success
                                         and abs(optimization_type * r.fun - test["optimum"]) < experiment_defaults["success_threshold"],
                               rs))
    success_rate  = success_total / float(len(rs))
    failures      = len(filter(lambda r: not r.success, rs))

    return (failures, success_rate, time_avg, nfev_avg)

def is_dm(optimizer):
    return optimizer["tag"] == "dm"

def conduct_experiment(exp_dir, test, optimizer):
    runs         = experiment_defaults["runs"]
    dimensions   = test["dimensions"] or sampler_defaults["dimensions"]
    range        = test["range"]      or sampler_defaults["range"]

    if is_dm(optimizer):
        logs_dir = path.join(exp_dir, "logs")
        os.makedirs(logs_dir)

    # construct the objective function from the string passed into the subprocess
    f_ = eval(test["function"]) # extract the function from the string
    optimization_type = test["optimization_type"] # a speedup
    f  = lambda x: optimization_type * f_(x) # handle maximization

    internal_optimizer = optimizer["optimizer"]
    optimizer_config   = optimizer["config"]
    rs = [] # collect the OptimizeResult objects in here.

    start_time = time()
    for i in xrange(runs):
        if is_dm(optimizer):
            optimizer_config["logfile"] = open(path.join(logs_dir, str(i) + ".log"), 'a')
        rs.append(internal_optimizer(f, dimensions, range, optimizer_config))
        if is_dm(optimizer):
            optimizer_config["logfile"].close()
    end_time = time()

    time_total = end_time - start_time

    return calculate_stats(test, rs, time_total) + (rs,)

def calculate_averages(statistics): # [[[a]]] -> [[a]]
    """ Take a [[[a]]], where [a] is the period sampling of a datum each iteration, [[a]] is such a sampling done on many individual runs,
        and finally [[[a]]] is a list of such statistics, and produce a [[a]] the same length as the input [[[a]]] that is a a list of averages
        w.r.t. time.
        """
    avgs = []
    for iteration in statistics:
        lasts = iteration[0]
        avgs.append([])
        for run in iteration:
            s = 0
            for (run_i, value) in enumerate(run):
                if value is None:
                    s += lasts[run_i]
                else:
                    s += value
                    lasts[run_i] = value
            avgs[-1].append(s/float(len(run)))

    avgs[-2][-1] *= len(filter(lambda x: x == 0, avgs[-2])) # For the performance. Zero looks like it's giving SA an advantage.
    return avgs

def write_experiment_data(exp_dir, complete_experiment):
    """ Expects a tuple in the form (name, averages, all runs), and writes out all the data for this experiment into the given directory. """
    for (name, average_vs_t, data) in complete_experiment:
        with open(exp_dir + "/" + name + ".txt", 'w') as f:
            for (i, run_i) in enumerate(data):
                f.write(str(average_vs_t[i]) if i < len(average_vs_t) else str(None))
                for value in run_i:
                    f.write(',' + str(value))
                f.write('\n')

def write_experiment_messages(exp_dir, rs):
    with open(exp_dir + '/' + "messages.txt", 'w') as f:
        map(lambda r: print(*tuple(r.message), sep='||', file=f), rs)

def experiment_task(args):
    edir, test, optimizer, names = args
    errprint("Begin experiment:", test["name"])
    # prepare the experiment directory
    exp_dir = edir + "/" + test["name"]
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    start_time = time()
    # Perform the experiment, rs is the actual OptimizeResult objects
    (failures, success_rate, time_avg, nfev_avg, rs) = conduct_experiment(exp_dir, test, optimizer);
    end_time = time()
    errprint(test["name"] + ":", "spent", end_time - start_time, "seconds performing experiment.")

    if is_dm(optimizer): # if the given optimizer is dm, we know how to inspect its internals and fish out useful information.
        start_time = time()
        # extract the vs list from each result; it contains those data that fluctuate over time. We transpose this list of lists to
        # line up all the data for a given iteration
        vs = map(lambda r: copy(r.opt.vs), rs)

        maxlen = reduce(max, imap(len, vs), 0)
        names_n = len(poll_names)
        data_vs_iter = [[[run[iteration_i][data_i] if iteration_i < len(run) else None for run in vs]
                                                                                       for iteration_i in xrange(maxlen)]
                                                                                       for data_i in xrange(names_n)]

        avgs = calculate_averages(data_vs_iter)
        complete_data = zip(names, avgs, data_vs_iter)

        # print the test-specific data to its own directory.
        write_experiment_data(exp_dir, complete_data)
        end_time = time()
        errprint(test["name"] + ":", "spent", end_time - start_time, "seconds writing.")

    write_experiment_messages(exp_dir, rs)
    errprint("End experiment:", test["name"])
    # the return value will get appended to the all_statistics of the master process
    return (test["name"], (success_rate, time_avg, nfev_avg, ndiv(nfev_avg, success_rate), failures))

# statistics measured: success rate, average runtime, average function evals, function value vs time, best minimum vs time, stepsize vs time
def conduct_all_experiments(edir, optimizer, experiment_defaults=experiment_defaults, names=poll_names):
    all_statistics = [] # we will collect the general statistics for each experiment here, to perform a global average over all experiments.
    global_statistics = []

    with open(path.join(edir, "in-progress.txt"), 'w') as f:
        print("A pipeline is running, with output to this folder.\n" +
              "DO NOT DELETE IT.\n" +
              "If the pipeline has stopped and this file is still present, then the pipeline probably crashed.",
              file=f)

    pool = mp.Pool(3)

    start_time = time()
    results = pool.map(experiment_task, izip(repeat(edir), tests, repeat(optimizer), repeat(names)))
    #results = [pool.apply_async(experiment_task, edir, test) for test in tests]
    #pool.close()
    #pool.join() # now we wait for the subprocesses to finish.

    # collect the results of the subprocesses
    all_statistics  = [result[1] for result in results]

    with open(path.join(edir, "averages.txt"), 'w', 1) as f:
        print_csv("test", "success rate", "average time", "average fun. evals.", "average performance", "failures", file=f)
        # print the general test data to the common file
        map(lambda (name, (success_rate, time_avg, nfev_avg, perf_avg, failures)): print_csv(name, success_rate, time_avg,
                                                                                             nfev_avg, perf_avg, failures, file=f),
            results)

        ## calculate the global statistics
        # transpose the list of statistics, and calculate the averages.
        global_statistics = tuple(map(lambda stat: sum(stat) / float(len(stat)), zip(*all_statistics)))
        # record the data
        print_csv("AVERAGE", *global_statistics, file=f)

    end_time = time()

    os.remove(path.join(edir, "in-progress.txt"))

    with open(path.join(edir, "time.txt"), 'w') as f:
        print(end_time - start_time, file=f)

    with open(path.join(edir, "optimizer.txt"), 'w') as f:
        print(optimizer["tag"], file=f)

# PIPELINE (called from run_test.py)
def run_test(edir, optimizer_name):
    return conduct_all_experiments(edir, optimizers[optimizer_name])

def parse_typical_poll_file(path):
    """ A poll file is one generated from write_experiment_data. The names of all the polls is in poll_names. """
    def sanitize(string):
        try:
            return float(string)
        except ValueError:
            return None

    with open(path) as f:
        csvtups = csv_to_tuples([line for line in f])
    average = []
    individuals = []
    i = 0
    for t in csvtups:
        average.append(sanitize(t[0]))
        indivs_at_i = map(sanitize, t[1:])
        individuals.append(indivs_at_i)
        i += 1
    return (average, individuals)

plots_config = {"individual_color":"0.6", "average_color":"blue"}

# PIPELINE
def generate_all_dm_plots(edir):
    dmdir = path.join(edir, "../dm")
    function_dirs = filter(lambda p: path.isdir(path.join(dmdir, p)), os.listdir(dmdir)) # function_dirs will be relative to dmdir

    if not function_dirs:
        raise ValueError("fatal: the previous step did not produce any data to plot.")

    for function in function_dirs:
        plot_dir = path.join(edir, function) # where to save the plot
        os.makedirs(plot_dir)
        for poll in poll_names:
            poll_pp = poll.replace("_", " ")
            data_path = path.join(dmdir, function, poll + ".txt")
            poll_data = parse_typical_poll_file(data_path)
            average, individuals = poll_data
            fig = plt.figure()
            fig.suptitle(" ".join([poll_pp, "vs. time"]))
            ax = fig.add_subplot(1,1,1)
            for individual_run in zip(*individuals):
                ys = list(takewhile(lambda v: v != None, individual_run))
                ax.plot(xrange(1, len(ys) + 1), ys, color=plots_config["individual_color"])
            ax.plot(xrange(1, len(average) + 1), average, color=plots_config["average_color"])
            fig.savefig(path.join(plot_dir, poll + ".pdf"))
            fig.clear()

# PIPELINE
def dm_plot_3d(edir, test_all_2d=False, show=False):
    # get all those functions whose domains are 2D, or everything if test_all_2d is true.
    tests_2d = filter(lambda fe: test_all_2d or fe["dimensions"] == 2, tests)

    opts = dict(dm_defaults)
    opts["verbosity"] = 0

    for test in tests_2d:
        f = eval(test["function"])
        while True:
            res = randomr_dm(f, 2, test["range"], opts)
            if res.success: # TODO THIS IS SO SKETCHY ...
                errprint("Completed", test["name"])
                break
        fig = plotf_3d(f, res.opt.lpos)
        if show:
            plt.show()


        fig.savefig(path.join(edir, test["name"] + ".pdf"))
