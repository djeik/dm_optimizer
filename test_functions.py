#!/usr/bin/env python

""" name: test_functions
    author: Jacob Errington
    date: 30 October 2014

    This module provides all the test functions to be used to test the
    optimizer. It reexports many functions from the DEAP project, and
    implements some more.

    The complete list of functions is the following.

    < write test functions here >

    A list of the functions is provided with the name "tests". Each entry in the
    list is a tuple of the following form.

        (name, function name, sign, dimensions, ranges, optimal value)

    Whereas the name is simply a human-readable form of the function name
    (without spaces, and usually all lowercase), the function name is such that
    getattr(test_functions, name) returns the function object that will
    evaluate the function at the given point.

    The sign is either 1 or -1, and should multiply the return value of the
    function object. This is to allow for maximization. This module
    additionally exports the global constants MINIMIZATION and MAXIMIZATION,
    equal to 1 and -1 respectively, to deal with this.

    Each function object takes an indexable iterator as input, and the required
    length of that iterator is given by the ``dimensions'' entry.

    The ranges entry specifies from what range should the initial guesses be
    taken for the minimization. If the paper describing the function did not
    provide information concerning the initial points, then the range entry is
    None, and the range can be chosen arbitrarily. This module exports a
    SAMPLER_DEFAULTS dictionary, whose entries ``dimensions'' and ``range''
    provide sensible defaults in case the dimensions or range are None. The
    number of ranges given must match the number of dimensions of the function.

    Finally, the optimum entry gives the ideal function value. This module
    exports a SUCCESS_EPSILON value that should be used as a margin of error
    for comparison with the final function value when optimizing.

"""

import numpy            as np
import dm_utils         as dmu
import jerrington_tools as jt
import deap.benchmarks  as bench
import sys

from itertools import imap, repeat

from math import pi, sin, cos, exp

SUCCESS_EPSILON = 0.001

SAMPLER_DEFAULTS = {"dimensions":5, "range":(-100, 100)}

MINIMIZATION =  1.0
MAXIMIZATION = -1.0

# This will create the unwrapped versions of the DEAP functions, and assign
# them in the current module.

for fun in ["ackley", "bohachevsky", "griewank", "h1", "rastrigin",
        "rosenbrock", "schaffer", "schwefel"]:
    setattr(sys.modules[__name__], fun, dmu.unwrap_bench(getattr(bench, fun)))

def _list_repeat(n, t):
    return list(repeat(t, n))

# These are the test entries for all the DEAP functions.

_ackley_test = ("ackley",
    "ackley",
    MINIMIZATION,
    2,
    _list_repeat(2, (-15, 30)),
    0)

_bohachevsky_test = ("bohachevsky",
    "bohachevsky",
    MINIMIZATION,
    7,
    _list_repeat(7, (-100, 100)),
    0)

_griewank_test = ("griewank",
    "griewank",
    MINIMIZATION,
    2,
    _list_repeat(2, (-600, 600)),
    0)

_h1_test = ("h1",
    "h1",
    MAXIMIZATION,
    4,
    _list_repeat(4, (-100, 100)),
    2)

_rastrigin_test = ("rastrigin",
    "rastrigin",
    MINIMIZATION,
    6,
    _list_repeat(6, (-5.12, 5.12)),
    0)

_rosenbrock_test = ("rosenbrock",
    "rosenbrock",
    MINIMIZATION,
    7,
    None,
    0)

_shaffer_test = ("schaffer",
    "schaffer",
    MINIMIZATION,
    4,
    _list_repeat(4, (-100, 100)),
    0)

_schwefel_test = ("schwefel",
    "schwefel",
    MINIMIZATION,
    4,
    _list_repeat(4, (-500, 500)),
    0)

def _sq(x):
    """ Prefix squaring. """
    return x**2

def _simon_f1(xy):
    """ A test function crafted by Simon. It's minimum value is zero at the
        origin.  It is only defined for two dimensions.
        """
    x, y = xy
    return 0.2 * (x**2 + y**2) / 2 + 10 * sin(x+y)**2 + \
            10 * sin(100 * (x - y))**2

def simonf2(xs):
    """ A variation on Simon's function. To prevent "origin-bias", which can be
        a problem with optimizers, we simply shift over the function, so that the
        minimum is now at (100, 100).
        """
    xy = xs - np.array([100, 100])
    return _simon_f1(xy)

_simonf2_test = ("simonf2",
    "simonf2",
    MINIMIZATION,
    2,
    _list_repeat(2, (0, 200)),
    0)

def braninRCOS(xs):
    """ (from [1])
        Dimensions: 2
        Range: (-5, 10), (0, 15)
        Type: minimization
        Optimum: 0.397887
        """
    x1, x2 = xs
    a = _sq((x2 - 5.1/(4.0 * _sq(pi)))*_sq(x1) + (5.0/pi)*x1 - 6)
    b = 10*(1 - (1/(8*pi))*cos(x1))
    return a + b + 10

_braninRCOS_test = ("braninRCOS",
    "braninRCOS",
    MINIMIZATION,
    2,
    [(-5, 10), (0, 15)],
    0.397887)

def easom(xs):
    """ (from [1])
        Dimensions: 2
        Range: (-10, 10), (-10, 10)
        Type: minimization
        Optimum: -1
        """
    x1, x2 = xs
    return -1.0 * cos(x1) * cos(x2) * exp(- _sq(x1 - pi) - _sq(x2 - pi))

_easom_test = ("easom",
    "easom",
    MINIMIZATION,
    2,
    _list_repeat(2, (-10, 10)),
    -1.0)

def goldsteinprice(xs):
    """ (from [1])
        Dimensions: 2
        Range: (-2, 2), (-2, 2)
        Optimum: 3
        """
    x1, x2 = xs
    u = 1 + _sq(x1 + x2 + 1) * (
            19 -
            14 * x1 +
            3  * _sq(x1) -
            14 * x2 +
            6 * x1 * x2 +
            3 * _sq(x2))
    v = 30 + _sq(2*x1 - 3 *x2) * (
            18 -
            32 * x1 +
            12 * _sq(x1) +
            48 * x2 -
            36 * x1 * x2 +
            27 * _sq(x2))
    return u * v

_goldsteinprice_test = ("goldsteinprice",
    "goldsteinprice",
    MINIMIZATION,
    2,
    _list_repeat(2, (-2, 2)),
    3.0)

def schubert(xs):
    """ (from [1])
        Dimensions: 2
        Range: (-10, 10), (-10, 10)
        Optimum: -186.7309
        """
    return reduce(
            op.mul,
            ((sum(j * cos((j + 1)*x + j) for j in xrange(1, 6)))
                                         for x in xs),
            1)

_schubert_test = ("schubert",
    "schubert",
    MINIMIZATION,
    2,
    _list_repeat(2, (-10, 10)),
    -186.7309)

def dejoung(xs):
    """ (from [1])
        Dimensions: 3 (implemented for n)
        Range: (-5, 5) for each
        Optimum: 0
        Note: this function is convex.
        """
    return sum(_sq(x) for x in xs)

_dejoung_test = ("dejoung",
    "dejoung",
    MINIMIZATION,
    3,
    _list_repeat(3, (-5, 5)),
    0.0)

def colville(xs):
    """ (from [1])
        Dimensions: 4
        Range: None
        Optimum: 0
        """
    x1, x2, x3, x4 = xs
    return (100  * _sq(_sq(x1) - x2) +
            _sq(x1 - 1) +
            _sq(x3 - 1) +
            90   * _sq(_sq(x3) - x4) +
            10.1 * (_sq(x2 - 1) + _sq(x4 - 1)) +
            19.8 * (x2 - 1) * (x4 - 1))

_colville_test = ("colville",
    "colville",
    MINIMIZATION,
    4,
    None,
    0)

def dixon(xs):
    """ (from [1])
        Dimensions: 10 (implemented for n)
        Range: None
        Optimum: 0
        """
    return (_sq(1 - xs[0]) + _sq(1 - xs[-1]) +
            sum(_sq(_sq(xs[i]) - xs[i+1]) for i in xrange(len(xs) - 1)))

_dixon_test = ("dixon",
    "dixon",
    MINIMIZATION,
    10,
    None,
    0)

def _zakharov_n(n):
    """ Construct a Zakharov function of ``n'' variables. """
    def z(xs):
        return sum(_sq(xs[i]) for i in xrange(n)) + \
                sum(_sq(0.5 * (i + 1) * xs[i]) for i in xrange(n)) + \
                sum(_sq(_sq(0.5 * (i + 1) * xs[i])) for i in xrange(n))
    return z

def _zakharov_n_test(n):
    """ Construct a test entry for the Zakharov function of ``n'' variables.
        """
    return ("zakharov" + str(n), "_".join(["zakharov", str(n)]), MINIMIZATION,
            n, _list_repeat(n, (-5, 10)), 0)

zakharov_2  = _zakharov_n(2)
zakharov_5  = _zakharov_n(5)
zakharov_10 = _zakharov_n(10)

_zakharov_2_test  = _zakharov_n_test(2)
_zakharov_5_test  = _zakharov_n_test(5)
_zakharov_10_test = _zakharov_n_test(10)


# Create the different ways of accessing the tests.

# Simply a list of the tests. People should use the tests_list and tests_dict
# objects instead.
_tests = [_ackley_test,
        _bohachevsky_test,
        _braninRCOS_test,
        _colville_test,
        _dejoung_test,
        _dixon_test,
        _easom_test,
        _goldsteinprice_test,
        _griewank_test,
        _h1_test,
        _rastrigin_test,
        _rosenbrock_test,
        _schwefel_test,
        _schubert_test,
        _shaffer_test,
        _simonf2_test,
        _zakharov_2_test,
        _zakharov_5_test,
        _zakharov_10_test]

# Convert each test into a dictionary, so that we have a list of dictionaries.
tests_list = map(
        lambda xs: dict(
            zip(["name", "function", "optimization_type",
                    "dimensions", "range", "optimum"],
                xs)),
        _tests)

# Convert the list into a dictionary, indexed by the name of the function.
tests_dict = dict(map(
        lambda x: (x["name"], x),
        tests_list))
