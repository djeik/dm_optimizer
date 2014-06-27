from __future__ import print_function

from scipy.optimize import fmin
import numpy as np
from numpy.linalg import norm
import random
import math
from math import sqrt, cos, sin, tan, exp
import deap.benchmarks as bench # various test functions
from copy import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sortedcontainers import sortedlist as sl

import sys


# These are the functions Simon defined to test it on:
def simon_f1(xy):
    x, y = xy
    return 0.2 * (x**2 + y**2) / 2 + 10 * sin(x+y)**2 + 10 * sin(100 * (x - y))**2

def simon_f2(xs):
    xy = xs - [100, 100]
    return simon_f1(xy)

def fst(xs):
    return xs[0]

def snd(xs):
    return xs[1]

# deap's benchmarking functions return the values as tuples with nothing in element 1,
# so we need to unpack element 0 which is the actual function-value.
def unwrapBench(f):
    return lambda x: f(x)[0]

# TODO replace this with sortedlists!
def mins(xs_, n, comp=lambda x: x):
    """ Find the n smallest values in some indexable collection, comparing by a given function. The default such function is the identity function."""
    xs = list(xs_) # copy xs
    nc = 0 # counter for the number of small values
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            if comp(xs[j]) < comp(xs[i]):
                t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
                nc += 1
                break
        if nc == n: break

    return xs[:n]

def randomGuess(dim, scale = 1):
    return np.array([scale * random.uniform(0,1) - scale/2.0 for _ in xrange(dim)])

class dm_optimizer:
    """ An optimizer using the difference map-based algorithm.
        Attributes:
            max_iterations      -- the maximum number of local minima to find before giving up.
            tolerance           -- the threshold above which we consider two points to be distinct.
            verbosity           -- controls debug output; higher numbers = more messages.
            first_target_ratio  -- determines how low the first target should be.
            greediness          -- how quickly should the target value be lowered.
            fun                 -- the objective function.
            pseudo              -- ?
            vals                -- the local minima collected along the way.
            lpos                -- the path the optimizer has followed along the objective function.

        Notes:
            vals is stored as a sorted list, since we frequently just need to get the n smallest values.
            It is a list of tuples (y, x), which is convenient since Python tuples are totally ordered according
            to their first element.
            """

    def __init__(self, fun, max_iter, tol=0.00001, verbosity=1, pseudo=0.00001, greediness=0.05, first_target_ratio=0.9):
        self.max_iterations     = max_iter
        self.tolerance          = tol
        self.verbosity          = verbosity
        self.first_target_ratio = first_target_ratio
        self.greediness         = greediness
        self.fun                = fun
        self.pseudo             = pseudo
        self.vals               = sl.SortedList()
        self.lpos               = []

    # A nice wrapper around the print function that will print to standard error only if the given priority is less
    # than the verbosity
    def logmsg(self, priority, *args, **kwargs):
        if priority <= self.verbosity:
            print(*args, file=sys.stderr, **kwargs)
        else:
            pass

    # Find the past minimum whose function-value is closest to the given one.
    def best_minimum_y(self, fv):
        u = sorted([x for x in self.vals], key=lambda (y, _): abs(y - fv))

        if len(u) < 2:
            return u[0]

        (y1, x1), (y2, x2) = u[:2]
        if abs(y1-fv) < self.tolerance: # if the *best* minimum is the *current* minimum (fixed-point like behaviour)...
            return (y2, x2) # return the *second*-best minimum.
        else:
            return (y1, x1)

    def best_minimum_x(self, pmin):
        u = sorted([x for x in self.vals], key=lambda (y, x): norm(pmin - x))

        if len(u) < 2:
            return u[0]

        (y1, x1), (y2, x2) = u[:2]
        if norm(x1 - pmin) < self.tolerance: # if the best minimum *is* the current minimum
            return (y2, x2) # then return the second-best minimum
        else:
            return (y1, x1)

    def new_target(self, best):
        # get the minimum y value recorded in our list of minima, and push down from there.
        self.target = best + self.greediness * (best - min(map((lambda v: v[0]), self.vals)))

    def refresh_target(self):
        # get the two best minima so far found
        bests = self.vals[:2] # remember, vals is sorted in ascending order of the y-value.
        # bests :: [(y,x)], too !
        self.logmsg(1, "Two best minima so far: ", bests[0], " and ", bests[1])
        diff = bests[0][0] - bests[1][0]
        if diff > 0:
            raise Exception("The difference between the two best minima should be negative!")
        self.logmsg(2, "Y-Difference between the minima: ", diff)
        self.target = bests[0][0] + self.greediness * diff

    def minimize(self, x0, x1):
        x0min = fmin(self.fun, x0, disp=self.verbosity==2)
        f0 = self.fun(x0min)

        self.vals.add( (f0, x0min) )
        self.target = self.first_target_ratio * f0

        nx1 = x1
        self.lpos = [(self.fun(nx1), nx1)]

        self.logmsg(2, "Entering loop.")

        try:
            for i in xrange(self.max_iterations):
                self.logmsg(2, "Guess point for this iteration:", nx1)
                pmin = fmin(self.fun, nx1, disp = self.verbosity == 2) # the x value at the local minimum for this iteration
                fv = self.fun(pmin) # the function value at the local minimum for this iteration
                delta = fv - self.target

                self.logmsg(2, "Got delta:", delta)

                if delta < 0: # we've fallen below the target, so we must update it.
                    self.new_target(fv) # to get the next target, all we need is the current function value and the list of minima so far
                    self.logmsg(1, "Target updated to", self.target)
                    delta = fv - self.target # recalculate delta for the new target
                    self.logmsg(2, "New delta:", delta)
                else:
                    self.logmsg(2, "Target y-value", self.target)

                # let's take the *best* minimum instead, i.e. the minimum closest to the target
                # wait that's no good. Simon's code has it taking the minimum closest to the current x-value!
                ynear, xnear = self.best_minimum_x(pmin)

                self.logmsg(2, "Nearest old minimum f", xnear, " = ", ynear, sep='')

                #if fv < ynear: #if the current local minimum is *better* than any previous one, we move *away* from the old one
                #    self.logmsg(1, "Doubling back!")
                #    stepdir = xnear - pmin
                #else: # otherwise, we move *towards* the old one
                stepdir = pmin - xnear

                deltay_prev = ynear - self.target # how far away is the best previously-found local minimum
                step = -delta / (self.pseudo if (delta - deltay_prev)**2 < self.pseudo**2 else (delta - deltay_prev)) * stepdir

                self.logmsg(2, "Step endpoints: ", pmin, " and ", xnear)

                nx1 += step # actually take the step
                self.lpos.append((self.fun(nx1), copy(nx1))) # add the new position to
                self.logmsg(2, "Took step ", step)

                self.vals.add( (fv, pmin) )

                if norm(step) < self.tolerance:
                    self.logmsg(1, "Found fixed point: f", pmin, " = ", ynear, sep='')
                    return pmin, fv, self.lpos
                elif norm(step) > 100:
                    self.logmsg(1, "Taking very large step!")
                    self.logmsg(1, "delta = ", delta, "; deltay_prev = ", deltay_prev, "; pmin = ", pmin, "; xnear = ", xnear, sep='')

                if i % 5 == 0: # A CONSTANT ! Consider changing this.
                    oldtarget = self.target
                    self.refresh_target() # mutates target
                    self.logmsg(1, "Refreshed target to", self.target)
                    # if ((oldtarget - target) / target)**2 > 0.01: # i.e. 0.1**2 # why?
        except:
            self.logmsg(-1, "An error occurred while optimizing.")
            raise

def test(f):
    optimizer = dm_optimizer(f, 2000, 0.000001)
    xmin, fv, lpos = optimizer.minimize(randomGuess(2, 2), randomGuess(2, 2))
    print("Found global minimum f", xmin, " = ", fv, sep='')
    print("Steps taken: ")
    map((lambda p: print(" ", p)), lpos)
    if len(xmin) == 2: #only plot if in 3D !
        plotf(f, lpos)

def testAckley():
    test(unwrapBench(bench.ackley))

# Visual debug tool for 3d
def plotf(f, xyzs_, start=np.array([-1,-1]), end=np.array([1,1]), smoothness=1.0, autobound=True, autosmooth=True):
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
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
