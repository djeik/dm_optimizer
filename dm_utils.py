#!/usr/bin/env python2

from __future__ import print_function

import sys
import random
import os

import dm_optimizer as dm
import numpy        as np

from scipy.optimize import basinhopping
from itertools      import repeat, imap, chain, islice, izip
from os             import path
from datetime       import datetime

import jerrington_tools as jt

def random_vector(dim=2, length=1):
    """ Construct a vector in a random direction with the given length. """
    ran_unit_ = np.random.sample(dim) - 0.5;
    return ran_unit_ / np.linalg.norm(ran_unit_) * length

# deap's benchmarking functions return the values as 1-tuples
# so we need to unpack element 0 which is the actual function-value.
def unwrap_bench(f):
    return lambda x: f(x)[0]

def intersperse(delimiter, seq):
        return islice(chain.from_iterable(izip(repeat(delimiter), seq)), 1, None)

def ipad_lists(padding, matrix):
    """ Pad the component lists of a list of lists to make it into a matrix. The operation is performed in-place, but the matrix is also returned,
        to allow chaining.
        """
    maxlen = reduce(max, imap(len, matrix), 0)
    for vector_ in matrix:
        vector = list(vector_) # copy the list
        vector.extend(repeat(padding, maxlen - len(vector)))
        yield vector

def imap_p(pred, true_f, false_f, seq):
    """ Lazily map a a predicate over a list, such that if the predicate is
        satisfied by an element x, then t(x) is appended called, else f(x) is
        called, where t and f are unary functions of the type of element in the
        original list.
        """
    return imap(lambda x: (true_f if pred(x) else false_f)(x), seq)

def pad_lists(*args):
    return list(ipad_lists(*args))

mkfprint = lambda f: lambda *args, **kwargs: print(*args, file=f, **kwargs)
errprint = mkfprint(sys.stderr)

# Construct a function that takes arbitrarily many positional and keyword arguments, but ignores them, always returning the same value.
const = lambda x: lambda *y, **kwargs: x
transpose = lambda x: zip(*x)

def nary2binary(f):
    """ Convert an n-ary function into a binary function, so that numpy
        vectorization can be applied properly. """
    return lambda x, y: f([x, y])

def make_experiment_dir(tag=""):
    """ Create a directory `results/[<tag>/]<datetime>` and return its path. """
    return jt.mkdir_p(path.join("results", tag, datetime.now().isoformat()))

def randomr_guess(rs):
    return np.array([random.uniform(*rr) for rr in rs])

def randomr_dm(f, d, ranges, dm_args):
    if d != len(ranges):
        raise ValueError("dimension does not match number of ranges given")
    return dm.minimize(f, randomr_guess(ranges), randomr_guess(ranges), **dm_args)

def randomr_sa(f, d, ranges, sa_args):
    r = basinhopping(f, randomr_guess(range), **sa_args)
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
    """ Expect a list of records, where each record consists of fields that are comma-separated.
        Return a list of lists. No handling of escaping the commas is done.
        """
    return [tuple(x.split(',')) for x in csv]

def print_csv(*args, **kwargs):
    print(*args, sep=',', **kwargs)

def ndiv(numerator, denominator, epsilon=1e-7):
    if isinstance(denominator, float):
        if denominator**2 < epsilon**2:
            return 0;
    elif isinstance(denominator, int):
        if denominator == 0:
            return 0;
    return numerator / float(denominator)

def is_dm(optimizer):
    """ Determine whether the given optimizer is the DM optimizer. """
    return optimizer["tag"] == "dm"

def normalize(v, tolerance=1e-8):
    """ Normalize the given vector. If its norm is below the tolerance (default 10^-8), then
        it cannot be properly normalized without introducing too much numerical error, so
        an exception is raised.
        """
    if norm(v) < tolerance:
        raise ValueError("Vector too small to properly normalize.")
    return v / norm(v)

def ndiv(numerator, denominator, epsilon=1e-7):
    """ Perform a "safe" division.

        If the denominator is a float and smaller than epsilon (default 10^-7),
        then the result is zero, rather than a very large number, or infinity, or an exception.
        If the denominator is an int and equal to zero, then the result is
        zero, rather than an exception.
        """
    if isinstance(denominator, float):
        if denominator**2 < epsilon**2:
            return 0;
    elif isinstance(denominator, int):
        if denominator == 0:
            return 0;
    return numerator / float(denominator)

def zipmap(f, seq):
    """ Map a function over a sequence and zip that sequence with the results of the applications. """
    return zip(seq, imap(f, seq))

def rebase_path(base_file, relative_file):
    return path.join(path.dirname(base_file), relative_file)

def compose(f, g):
    return lambda *args, **kwargs: f(g(*args, **kwargs))

curry2 = lambda f: lambda x: lambda y: f(x, y)
uncurry2 = lambda f: lambda x, y: f(x)(y)
curry3 = lambda f: lambda x: lambda y: lambda z: f(x, y, z)
map_c = curry2(map)
imap_c = curry2(imap)

# transform a function of many arguments into a function that takes one tuple
splat = lambda f: lambda args: f(*args)

# transform a function of one collection into a function of many arguments
unsplat = lambda f: lambda *args: f(args)

# The (unary) identity function
nop = lambda x: x

# The n-ary identity function
nops = unsplat(nop)

# LAW: compose(splat, unsplat) = nop

map_z = lambda f: lambda *args: map(splat(f), zip(*args))

# from a given key or index, make a function that projects the associated value from a dict of indexable.
project = lambda k: lambda d: d[k]

flip = lambda f: lambda x, y: f(y, x)
