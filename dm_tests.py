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

from itertools import repeat, imap, ifilter, islice, chain, izip

import dm_optimizer as dm
from dm_optimizer import dm_optimizer

import multiprocessing as mp

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) # unbuffered output

# These are the functions Simon defined to test it on:
def simon_f1(xy):
    x, y = xy
    return 0.2 * (x**2 + y**2) / 2 + 10 * sin(x+y)**2 + 10 * sin(100 * (x - y))**2

def simon_f2(xs):
    xy = xs - np.array([100, 100])
    return simon_f1(xy)

# deap's benchmarking functions return the values as 1-tuples
# so we need to unpack element 0 which is the actual function-value.
def unwrap_bench(f):
    return lambda x: f(x)[0]

def random_guess(dim, scale = 1):
    return np.array([scale * (random.uniform(0,1) - 1/2.0) for _ in xrange(dim)])

def randomr_guess(dim, r=(-1,1)):
    return np.array([random.uniform(*rr) for rr in repeat(r, dim)])

def intersperse(delimiter, seq):
        return islice(chain.from_iterable(izip(repeat(delimiter), seq)), 1, None)

def ipad_lists(padding, matrix):
    """ Pad the component lists of a list of lists to make it into a matrix. The operation is performed in-place, but the matrix is also returned,
        to allow chaining.
        """
    maxlen = reduce(max, imap(len, matrix))
    for vector_ in matrix:
        vector = list(vector_) # copy the list
        vector.extend(repeat(padding, maxlen - len(vector)))
        yield vector

def pad_lists(*args):
    return list(ipad_lists(*args))

def randomr_dm(f, d, range, dm_args={}):
    return dm.minimize(f, randomr_guess(d, range), randomr_guess(d, range), **dm_args)

def randomr_sa(f, d, range, sa_args={}):
    r = basinhopping(f, randomr_guess(d, range), **sa_args)
    r.success = True # TODO this needs to be something that sucks less.
    return r

def read_2d_csv(filename):
    dats = []
    with open(filename) as f:
        for line in f:
            dats.append(tuple(map(float, line.split(','))))
    return zip(*dats)

def write_2d_csv(fname, dats):
    with open(fname, 'w') as f:
        for (iter_count, success_rate) in dats:
            print(iter_count, success_rate, sep=',', file=f)

def tuples_to_csv(dats):
    return '\n'.join([','.join(map(str, x)) for x in dats])

def csv_to_tuples(csv):
    return [tuple(x.split(',')) for x in csv.split('\n')]

def print_csv(*args, **kwargs):
    print(*args, sep=',', **kwargs)

def write_str(fname, string):
    with open(fname, 'w') as f:
        f.write(string)

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

minimization = 1
maximization = -1

tests = map(lambda xs: dict(zip(
        ["name",        "function",                        "optimization_type", "dimensions", "range",      "optimum"], xs)),
        [("ackley",     "unwrap_bench(bench.ackley)",      minimization,        None,         (-15, 30),     0),
        ("cigar",       "unwrap_bench(bench.cigar)",       minimization,        None,         None,          0),
        ("sphere",      "unwrap_bench(bench.sphere)",      minimization,        None,         None,          0),
        ("bohachevsky", "unwrap_bench(bench.bohachevsky)", minimization,        None,         (-100, 100),   0),
        ("griewank",    "unwrap_bench(bench.griewank)",    minimization,        None,         (-600, 600),   0),
        ("h1",          "unwrap_bench(bench.h1)",          maximization,        2,            (-100, 100),   2),
        ("himmelblau",  "unwrap_bench(bench.himmelblau)",  minimization,        2,            (-6, 6),       0),
        ("rastrigin",   "unwrap_bench(bench.rastrigin)",   minimization,        None,         (-5.12, 5.12), 0),
        ("rosenbrock",  "unwrap_bench(bench.rosenbrock)",  minimization,        None,         None,          0),
        ("schaffer",    "unwrap_bench(bench.schaffer)",    minimization,        None,         (-100, 100),   0),
        ("schwefel",    "unwrap_bench(bench.schwefel)",    minimization,        None,         (-500, 500),   0),
        ("simon_f2",    "simon_f2",                        minimization,        2,            (-100, 100),   0)])
        ## The following functions are 'weird' in some way that makes testing too difficult.
        #(unwrap_bench(bench.rastrigin_scaled),
        #(bench.rastrigin_skew
        #(unwrap_bench(bench.rand), None, None, None, None),
        #(unwrap_bench(bench.plane), minimization, None, None, 0),

# we're going to stop naming things after dates, due to the new commit-enforcing policy.
def dm_poll_callback(self):
    self.vs.append( (self.fv, self.vals[0].y, norm(self.step)) )
poll_names = ["function_value", "best_minimum", "step_size"] # the names of the things extracted from the optimizer internal state

sampler_defaults = {"dimensions":5, "range":(-100, 100)}
experiment_defaults = {"runs":250, "success_threshold":0.001}
dm_defaults = {"refresh_rate":4, "max_iterations":250, "callback":dm_poll_callback}
sa_defaults = {"niter":100}

optimizers = {"dm":{"tag":"dm", "optimizer":randomr_dm, "config":dm_defaults},
              "sa":{"tag":"sa", "optimizer":randomr_sa, "config":sa_defaults}}

def conduct_experiment(test, optimizer):
    runs         = experiment_defaults["runs"]
    dimensions   = test["dimensions"] or sampler_defaults["dimensions"]
    range        = test["range"]      or sampler_defaults["range"]

    # construct the objective function from the string passed into the subprocess
    f_ = eval(test["function"]) # extract the function from the string
    optimization_type = test["optimization_type"] # a speedup
    f  = lambda x: optimization_type * f_(x) # handle maximization

    internal_optimizer = optimizer["optimizer"]
    optimizer_config   = optimizer["config"]
    rs = [] # collect the OptimizeResult objects in here.
    start_time = time()
    for i in xrange(runs):
        rs.append(internal_optimizer(f, dimensions, range, optimizer_config))
    end_time   = time()

    time_total    = end_time - start_time
    time_avg      = time_total / float(runs)
    nfev_total    = sum(imap(lambda r: r.nfev, rs))
    nfev_avg      = nfev_total / float(runs)
    success_total = len(filter(lambda r: r.success and abs(r.fun - test["optimum"]) < experiment_defaults["success_threshold"], rs))
    success_rate  = success_total / float(runs)
    failures      = len(filter(lambda r: not r.success, rs))

    return (failures, success_rate, time_avg, nfev_avg, rs)

def calculate_averages(statistics): # [[[a]]] -> [[a]]
    """ Take a [[[a]]], where [a] is the period sampling of a datum each iteration, [[a]] is such a sampling done on many individual runs,
        and finally [[[a]]] is a list of such statistics, and produce a [[a]] the same length as the input [[[a]]] that is a a list of averages
        w.r.t. time.
        """
    def avg_vs_t(s):
# take in the list of runs and their courses, and line them up
        s_ = zip(*pad_lists(0, s)) # the first element of this thing is the list of values of the variable at the first iteration
        # for each list in s_, we compute the average, making a list of averages. That is the progression of the average value over time.
        return map(lambda x: sum(x) / float(len(x)), s_)
    return map(avg_vs_t, statistics)

def write_experiment_data(exp_dir, complete_experiment):
    """ Expects a tuple in the form (name, averages, all runs), and writes out all the data for this experiment into the given directory. """
    for (name, average_vs_t, data) in complete_experiment:
        with open(exp_dir + "/" + name + ".txt", 'w') as f:
            for (i, run_i) in enumerate(zip(*pad_lists(None, data))):
                #f.write(str(i) + ',') # writing the iteration number is not useful, since they're in order. Therefore, line # = iteration #.
                f.write(str(average_vs_t[i]))
                for value in run_i:
                    f.write(',' + str(value))
                f.write('\n')

def write_experiment_messages(exp_dir, rs):
    with open(exp_dir + '/' + "messages.txt", 'w') as f:
        map(lambda r: print(*tuple(r.message), sep='||', file=f), rs)

def experiment_task(args):
    edir, test, optimizer, names = args
    print("Begin experiment:", test["name"])
    # prepare the experiment directory
    exp_dir = edir + "/" + test["name"]
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Perform the experiment, rs is the actual OptimizeResult objects
    (failures, success_rate, time_avg, nfev_avg, rs) = conduct_experiment(test, optimizer);

    if optimizer["tag"] == "dm": # if the given optimizer is dm, we know how to inspect its internals and fish out useful information.
        # extract the vs list from each result; it contains those data that fluctuate over time. We transpose this list of lists to
        # line up all the data for a given iteration
        padded_lists = pad_lists([], map(lambda r: copy(r.opt.vs), rs))
        test_data = zip(*padded_lists) # [([a],[a],[a])] -> ([[a]], [[b]], [[c]])

        avgs = calculate_averages(test_data)
        complete_data = zip(names, avgs, test_data)

        #import pdb; pdb.set_trace()
        # print the test-specific data to its own directory.
        write_experiment_data(exp_dir, complete_data)

    write_experiment_messages(exp_dir, rs)
    print("End experiment:", test["name"])
    return (test["name"], (success_rate, time_avg, nfev_avg, failures)) # this will get appended to the all_statistics of the master process

# statistics measured: success rate, average runtime, average function evals, function value vs time, best minimum vs time, stepsize vs time
def conduct_all_experiments(edir, optimizer, experiment_defaults=experiment_defaults, names=poll_names):
    all_statistics = [] # we will collect the general statistics for each experiment here, to perform a global average over all experiments.
    global_statistics = []

    pool = mp.Pool()

    start_time = time()
    results = pool.map(experiment_task, izip(repeat(edir), tests, repeat(optimizer), repeat(names)))
    #results = [pool.apply_async(experiment_task, edir, test) for test in tests]
    #pool.close()
    #pool.join() # now we wait for the subprocesses to finish.

    # collect the results of the subprocesses
    all_statistics  = [result[1] for result in results]

    with open(path.join(edir, "averages.txt"), 'w', 1) as f:
        print_csv("test", "success rate", "average time", "average fun. evals.", "failures", file=f)
        # print the general test data to the common file
        map(lambda (name, (success_rate, time_avg, nfev_avg, failures)): print_csv(name, success_rate, time_avg, nfev_avg, failures, file=f),
                results)

        ## calculate the global statistics
        # transpose the list of statistics, and calculate the averages.
        global_statistics = tuple(map(lambda stat: sum(stat) / float(len(stat)), zip(*all_statistics)))
        # record the data
        print_csv("AVERAGE", *global_statistics, file=f)
        end_time = time()

    with open(path.join(edir, "time.txt"), 'w') as f:
        print(end_time - start_time, file=f)

    with open(path.join(edir, "optimizer.txt"), 'w') as f:
        print(optimizer["tag"], file=f)

def run_test(edir, optimizer_name):
    return conduct_all_experiments(edir, optimizers[optimizer_name])

def parse_typical_poll_file(path):
    """ A poll file is one generated from write_experiment_data. The names of all the polls is in poll_names. """
    with open(path) as f:
        csvtups = csv_to_tuples([line for line in f])
    average = []
    individuals = []
    for t in csvtups:
        average.append(t[0])
        individuals.append(islice(t, 1))
    return (average, individuals)

def generate_all_dm_plots(edir):
    dmdir = path.join(edir, "../dm")
    function_dirs = filter(path.isdir, os.listdir(dmdir)) # function_dirs will be relative to dmdir

    for function in function_dirs:
        plot_dir = path.join(edir, function)
        os.makedirs(plot_dir)
        for poll in poll_names:
            data_path = path.join(dmdir, function, poll + ".txt")
            poll_data = parse_typical_poll_file(data_path)
            average, individuals = poll_data
            fig = plt.figure()
            ax = fig.add_axes()
            ax.plot(xrange(1, len(average) + 1), average)
            ax.plot(xrange(1, len(individuals) + 1), *zip(*individuals))
            fig.savefig(path.join(plot_dir, poll + ".pdf"))



