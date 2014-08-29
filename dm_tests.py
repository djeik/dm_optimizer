#!/usr/bin/env python

from __future__ import print_function

# System and standard libraries
import sys
import random
import os
import multiprocessing as mp
from time           import time
from math           import sin
from copy           import copy
from os             import path
from itertools      import repeat, imap, ifilter, islice, chain, izip, takewhile
from datetime       import datetime
from matplotlib     import cm

# The main scientific and numeric libraries
import numpy as np
from numpy.linalg   import norm
from scipy.optimize import basinhopping
import deap.benchmarks as bench # various test functions
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Other modules in this project
import dm_optimizer as dm
from dm_optimizer   import dm_optimizer

from dm_tests_config import *
from dm_utils        import *

import jerrington_tools as j

# Force line buffered output regardless of the type of device connected to stdout
# This is necessary to avoid block buffering that becomes enabled by default by the C library's stdio
# if it detects that a terminal is connected to stdout.
sys.stdout = os.fdopen(sys.__stdout__.fileno(), 'w', 1)

def simon_f1(xy):
    """ A test function crafted by Simon. It's minimum value is zero at the origin.
        It is only defined for two dimensions.
        """
    x, y = xy
    return 0.2 * (x**2 + y**2) / 2 + 10 * sin(x+y)**2 + 10 * sin(100 * (x - y))**2

def simonf2(xs):
    """ A variation on Simon's function. To prevent "origin-bias", which can be a problem with optimizers,
        we simply shift over the function, so that the minimum is now at (100, 100).
        """
    xy = xs - np.array([100, 100])
    return simon_f1(xy)

def plotf_3d(f, xyzs_, start=np.array([-1,-1]), end=np.array([1,1]), smoothness=1.0, autobound=True, autosmooth=True):
    """ Plot a function in three dimensions and show a trajectory on it.

        Arguments:
            f (function)       -- the function to plot.
            xyzs_ (ndarray)    -- the trajectory to draw, in the format [(z, (x, y)].
            start (ndarray)    -- a corner in the bounding box of the sampled domain.
            end (ndarray)      -- the opposite corner in the bounding box.
            smoothness (float) -- how sparsely is the domain sampled. Smaller values look better, but take
                                  longer to render.
            autobound (bool)   -- automatically determine the bounding box based on the trajectory if True.
            autosmooth (bool)  -- automatically calculate a decent smoothing value, based on the bounding box.

        Return:
            The generated figure, so that it may be rendered to a window or saved to a file by the caller.
        """
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

def conduct_experiment(exp_dir, test, optimizer, experiment_defaults=experiment_defaults):
    runs         = experiment_defaults["runs"]
    dimensions   = test["dimensions"] or sampler_defaults["dimensions"]
    range        = test["range"]      or sampler_defaults["range"]

    if is_dm(optimizer):
        logs_dir = path.join(exp_dir, "logs")
        mkdir_p(logs_dir)

    # construct the objective function from the string passed into the subprocess
    f_ = eval(test["function"]) # extract the function from the string
    optimization_type = test["optimization_type"] # a speedup
    f  = lambda x: optimization_type * f_(x) # handle maximization

    internal_optimizer = optimizer["optimizer"]
    rs = [] # collect the OptimizeResult objects in here.

    start_time = time()
    for i in xrange(runs):
        optimizer_config = optimizer["config_gen"]()
        if is_dm(optimizer):
            optimizer_config["logfile"] = open(path.join(logs_dir, str(i) + ".log"), 'a')
        # each run needs to gen its own config since the callbacks are objects whose data cannot be shared among multiple runs
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
    j.errprint("Begin experiment:", test["name"])
    # prepare the experiment directory
    exp_dir = edir + "/" + test["name"]
    mkdir_p(exp_dir)

    start_time = time()
    # Perform the experiment, rs is the actual OptimizeResult objects
    (failures, success_rate, time_avg, nfev_avg, rs) = conduct_experiment(exp_dir, test, optimizer);
    end_time = time()
    j.errprint(test["name"] + ":", "spent", end_time - start_time, "seconds performing experiment.")

    if is_dm(optimizer): # if the given optimizer is dm, we know how to inspect its internals and fish out useful information.
        start_time = time()
        # extract the vs list from each result; it contains those data that fluctuate over time. We transpose this list of lists to
        # line up all the data for a given iteration
        vs = map(lambda r: copy(r.opt.callback.vs), rs)

        maxlen       = reduce(max, imap(len, vs), 0)
        names_n      = len(poll_names)
        data_vs_iter = [[[run[iteration_i][data_i] if iteration_i < len(run) else None for run in vs]
                                                                                       for iteration_i in xrange(maxlen)]
                                                                                       for data_i in xrange(names_n)]

        avgs          = calculate_averages(data_vs_iter)
        complete_data = zip(names, avgs, data_vs_iter)

        # print the test-specific data to its own directory.
        write_experiment_data(exp_dir, complete_data)
        end_time = time()
        j.errprint(test["name"] + ":", "spent", end_time - start_time, "seconds writing.")

    write_experiment_messages(exp_dir, rs)
    j.errprint("End experiment:", test["name"])
    # the return value will get appended to the all_statistics of the master process
    return (test["name"], (success_rate, time_avg, nfev_avg, ndiv(nfev_avg, success_rate), failures))

def conduct_all_experiments_inner(edir, optimizer, names=poll_names, subproc_count=4):
    """ Inner function called by conduct_all_experiments that runs a given
        number of subprocesses to run each of the test objective functions
        listed in dm_tests_config.py, with a given set of experiment settings,
        taken by default from experiment_defaults.  Internally, this function
        calls experiment_task for each of the tests, which will generate a
        folder with the name of the test in the "edir" directory, where any
        information relevant to the function will be saved.
        """
    pool = mp.Pool(subproc_count)

    results = pool.map(experiment_task, izip(repeat(edir), tests, repeat(optimizer), repeat(names)))

    # collect the results of the subprocesses
    all_statistics  = [result[1] for result in results]

    ## calculate the global statistics
    # transpose the list of statistics, and calculate the averages.
    global_averages = tuple(map(lambda stat: sum(stat) / float(len(stat)), zip(*all_statistics)))
    global_stdevs   = tuple(map(np.std, zip(*all_statistics)))

    return (results, all_statistics, global_averages, global_stdevs)

# statistics measured: success rate, average runtime, average function evals, function value vs time, best minimum vs time, stepsize vs time
def conduct_all_experiments(edir, optimizer, experiment_defaults=experiment_defaults, names=poll_names):
    all_statistics = [] # we will collect the general statistics for each experiment here, to perform a global average over all experiments.
    global_statistics = []

    with open(path.join(edir, "in-progress.txt"), 'w') as f:
        print("A pipeline is running, with output to this folder.\n" +
              "DO NOT DELETE IT.\n" +
              "If the pipeline has stopped and this file is still present, then the pipeline probably crashed.",
              file=f)

    start_time = time()
    results, all_statistics, global_averages, global_stdevs = conduct_all_experiments_inner(edir, optimizer, names)
    end_time = time()

    with open(path.join(edir, "averages.txt"), 'w', 1) as f:
        print_csv("test", "success rate", "average time", "average fun. evals.", "average performance", "failures", file=f)
        # print the general test data to the common file
        map(lambda (name, (success_rate, time_avg, nfev_avg, perf_avg, failures)): print_csv(name, success_rate, time_avg,
                                                                                             nfev_avg, perf_avg, failures, file=f),
            results)

        # record the data
        print_csv("AVERAGE", *global_averages, file=f)
        print_csv("STDEV", *global_stdevs, file=f)

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

# PIPELINE
def generate_all_dm_plots(edir):
    dmdir = path.join(edir, "../dm")
    function_dirs = filter(lambda p: path.isdir(path.join(dmdir, p)), os.listdir(dmdir)) # function_dirs will be relative to dmdir

    if not function_dirs:
        raise ValueError("fatal: the previous step did not produce any data to plot.")

    for function in function_dirs:
        plot_dir = path.join(edir, function) # where to save the plot
        mkdir_p(plot_dir)
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
                j.errprint("Completed", test["name"])
                break
        fig = plotf_3d(f, res.opt.lpos)
        if show:
            plt.show()

        fig.savefig(path.join(edir, test["name"] + ".pdf"))

def safe_set_iteration_count(optimizer, iterations_count):
    """ For the given optimizer, generate a new configuration dictionary with the number of iterations set
        to the given value. The new dictionary is returned.
        """
    is_optimizer = lambda x: optimizer["tag"] == x
    if is_optimizer("dm"):
        optimizer["config"]["max_iterations"] = iterations_count
    elif is_optimizer("sa"):
        optimizer["config"]["niter"] = iterations_count
    else:
        raise ValueError("Unrecognized optimizer: %s." % optimizer["tag"])

def solved_vs_iterations_inner_inner(args):
    run_number, test, test_dir, optimizer_name, extra_optimizer_config = args

    my_optimizer = optimizer_config_gen(dict(optimizers[optimizer_name]), test["optimum"], extra_optimizer_config)
    safe_set_iteration_count(my_optimizer, iterations_config["end"])

    output_dir = path.join(test_dir, str(run_number))
    experiment_output = my_optimizer["optimizer"](eval(test["function"]), test["dimensions"],
                                                  test["range"] or sampler_defaults["range"],
                                                  my_optimizer["config"])

    # since we record one v for each iter, and the optimizer will end if the global minimum is found, the length of the vs
    # represents how many iterations it took to find the global minimum
    v = len(my_optimizer["config"]["callback"].vs)
    j.errprint("Run #", run_number, ": ", v, sep='')
    return v

def solved_vs_iterations_inner(solver_dir, optimizer_name, test,
        extra_optimizer_config, experiment_settings=experiment_defaults):
    """ Run the given optimizer on the given test, to generate data for a "fraction of successful runs versus time"
        plot.

        Arguments:
            solver_dir (string)           -- the directory in which to create the data/ and results/ directories, where
                                             runtime information about the solver and the resulting <function>.csv file are
                                             saved, respectively. Usually, `solver_dir` is something like
                                             "results/<date & time>/<solver name>".
            optimizer_name (string)       -- the name of the solver to use. This is used as a key in the `optimizers` dict
                                             declared in the dm_tests_config module.
            test (dict)                   -- the test to run the optimizer on. This should be an entry from the `tests`
                                             list declared in the dm_tests_config module.
            extra_optimizer_config (dict) -- extra configuration dict to pass to the solver.

        Returns:
            A list whose value at index `i` is the number of unfinished runs still going at iteration i.
        """
    mkdir_p(solver_dir)

    #test_dir is date/optimizer/
    test_dir = path.join(solver_dir, "data", test["name"])
    mkdir_p(test_dir)
    result_dir = path.join(solver_dir, "results")

    j.errprint("Running test: ", test["name"], "...", sep='')

    pool = mp.Pool(solved_vs_iterations_subproc_count)

    data_points = pool.map(
            solved_vs_iterations_inner_inner,
            izip(xrange(experiment_settings["runs"]),
                 repeat(test),
                 repeat(test_dir),
                 repeat(optimizer_name),
                 repeat(extra_optimizer_config)))

    j.errprint("Done running test.")

    j.errprint("Parsing experiment data... ", end='')

    # data_points is just a list of ints, that say how long it took for that run to finish
    def alive_vs_t(lifetimes):
        ls = list(lifetimes)
        alives = []
        for i in xrange(iterations_config["end"]):
            # remove all the lifetimes that are less that the iteration number
            ls = filter(lambda n: n > i, ls)
            alives.append(len(ls)) # record how many runs were still going at this iteration
        return alives

    alives_vs_t = alive_vs_t(data_points)

    j.errprint("Done.")

    return alives_vs_t

def parse_solved_vs_iterations_data_for_one_optimizer(data_dir, runs_count):
    """ Return a dict associating each objective function in dm_tests_config.tests to a list that represents
        the fraction of solved runs versus time.

        Arguments:
            data_dir (string) -- the path to the folder that contains the <function>.csv files.
            runs_count (int)  -- the number of runs to average over.

        Notes:
            The <function>.csv files should be the number of runs _still operating_ at the iteration number
        identified by the line number in the file. This function will take care of performing the calculation
                (a - x) / a
            where a is the total number of runs (runs_count) and x is the value on that line.
        """
    path_to_data = lambda func_name: path.join(data_dir, func_name + ".csv")
    to_fraction = lambda x: (runs_count - x) / float(runs_count)

    test_names = map(project("name"), tests)

    d = dict(zipmap(lambda test_name: with_file(
            j.map_c(j.compose(to_fraction, int)), # make a func that takes a seq, conv each elm to int and frac
            path_to_data(test_name)), # for each test name we get the path to the data file
        test_names))

    assert(len(d.keys()) == len(test_names) and len(test_names) == len(tests))

    return d

def parse_solved_vs_iterations_data(data_dir, runs_count):
    optimizer_names = optimizers.keys()
    path_to_data = lambda solver_name, func_name: path.join(
            data_dir, solver_name, "results", func_name + ".csv")

    data = dict(map(
        lambda test: (test["name"], dict(map(
            lambda optimizer: (optimizer, with_file(
                lambda f: map(
                    lambda x: (runs_count - int(x)) / float(runs_count),
                    f),
                path_to_data(optimizer, test["name"]))),
            optimizer_names))),
        tests)) # :: Map FunctionName ([FractionSolved1], [FractionSolved2])
    return data

def solved_vs_iterations_plots_pro(path_to_data,
        iterations_count=iterations_config["end"], runs_count=experiment_defaults["runs"]):
    # we have data per stepsize stored in all-dm, in folders named with the stepsize.
    path_to_stepsizes = path.join(path_to_data, "dm")
    stepsizes = imap(
            lambda s: (float(s), path.join(path_to_stepsizes, s, "dm", "results")),
            os.listdir(path_to_stepsizes))

    # make the directory where we'll save the plots
    plot_dir = path.join(path_to_data, "plots")
    mkdir_p(plot_dir)

    # dict like D[stepsize][function_name] : list representing fraction completed by time
    functions_by_stepsize = map(
        lambda (stepsize, directory): (stepsize,
                                       parse_solved_vs_iterations_data_for_one_optimizer(directory, runs_count)),
        stepsizes)

    # now we need to do the same but for SA
    simulated_annealing_path = path.join(path_to_data, "sa", "sa", "results") # here's the successful run of SA

    sa_data = parse_solved_vs_iterations_data_for_one_optimizer(simulated_annealing_path, runs_count)

    for test in tests:
        if not (test["name"] in sa_data and all(imap(lambda d: test["name"] in d, imap(project(1), functions_by_stepsize)))):
            print("Skipping function", test["name"], "because one or more datasets don't have entries for it.")
            continue
        fig = plt.figure()
        fig.suptitle("%s (%d dimensions) - fraction of successful runs vs. time" % (test["name"], test["dimensions"]))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0, iterations_count)
        ax.set_ylim(0, 1)
        ax.plot(sa_data[test["name"]], label="sa", color=(0, 0, 1))
        for (size, dm_data) in functions_by_stepsize:
            j.errprint("Plotting for size", size)
            ax.plot(dm_data[test["name"]], label="dm %f" % size, color=(size/1.2, 0.5, 0.5))
        #ax.legend()
        fig.savefig(path.join(plot_dir, test["name"] + ".eps"))

def solved_vs_iterations_plots(data_dir,
        iterations_count=iterations_config["end"], runs_count=experiment_defaults["runs"]):
    data = parse_solved_vs_iterations_data(data_dir, runs_count)

    mkdir_p(path.join(data_dir, "solved_vs_iterations"))

    for (func_name, each_optimizer) in data.items():
        fig = plt.figure()
        fig.suptitle("%s - fraction of successful runs vs. time" % func_name)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(0, iterations_count)
        ax.set_ylim(0, 1)
        for (solver_name, values) in each_optimizer.items():
            ax.plot(values, label=solver_name)
        ax.legend()
        fig.savefig(path.join(data_dir, "solved_vs_iterations", func_name + ".pdf"))


