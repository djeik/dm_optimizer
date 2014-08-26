#!/usr/bin/env python2

from __future__ import print_function
import sys
import random
from scipy.optimize import basinhopping
import dm_optimizer as dm
import numpy as np
from itertools import repeat, imap, chain, islice, izip
import os
from os import path

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

imap_p = lambda p, t, f, xs: imap(lambda x: t(x) if p(x) else f(x), xs)

def pad_lists(*args):
    return list(ipad_lists(*args))

mkfprint = lambda f: lambda *args, **kwargs: print(*args, file=f, **kwargs)
errprint = mkfprint(sys.stderr)

# Construct a function that takes arbitrarily many positional and keyword arguments, but ignores them, always returning the same value.
const = lambda x: lambda *y, **kwargs: x
transpose = lambda x: zip(*x)

def randomr_guess(dim, r=(-1,1)):
    return np.array([random.uniform(*rr) for rr in repeat(r, dim)])

def randomr_dm(f, d, range, dm_args):
    return dm.minimize(f, randomr_guess(d, range), randomr_guess(d, range), **dm_args)

def randomr_sa(f, d, range, sa_args):
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
    """ Expect a list of records, where each record consists of fields that are comma-separated.
        Return a list of lists. No handling of escaping the commas is done.
        """
    return [tuple(x.split(',')) for x in csv]

def print_csv(*args, **kwargs):
    print(*args, sep=',', **kwargs)

def is_dm(optimizer):
    return optimizer["tag"] == "dm"

def normalize(v, tolerance=1e-8):
    if norm(v) < tolerance:
        raise ValueError("Vector too small to properly normalize.")
    return v / norm(v)

def ndiv(numerator, denominator, epsilon=1e-7):
    if isinstance(denominator, float):
        if denominator**2 < epsilon**2:
            return 0;
    elif isinstance(denominator, int):
        if denominator == 0:
            return 0;
    return numerator / float(denominator)

def with_file(f, path, mode='r'):
    """ Run a function on a file handle, using a with-statement. Useful in lambdas. """
    with open(path, mode) as handle:
        return f(handle)

def zipmap(f, seq):
    """ Map a function over a sequence and zip that sequence with the results of the applications. """
    return zip(seq, imap(f, seq))

def rebase_path(base_file, relative_file):
    return path.join(path.dirname(base_file), relative_file)

def mkdir_p(dir):
    if not path.exists(dir):
        os.makedirs(dir)
        return True
    else:
        return False

def compose(f, g):
    return lambda *args, **kwargs: f(g(*args, **kwargs))

curry2 = lambda f: lambda x: lambda y: f(x, y)
uncurry2 = lambda f: lambda x, y: f(x)(y)
curry3 = lambda f: lambda x: lambda y: lambda z: f(x, y, z)
map_c = curry2(map)
imap_c = curry2(imap)

# transform a function of many arguments into a function that takes one tuple
splat = lambda f: lambda args: f(*args)

map_z = lambda f: lambda *args: map(splat(f), zip(*args))

# from a given key or index, make a function that projects the associated value from a dict of indexable.
project = lambda k: lambda d: d[k]

flip = lambda f: lambda x, y: f(y, x)
