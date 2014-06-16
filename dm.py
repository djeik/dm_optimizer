from __future__ import print_function

from scipy.optimize import fmin
import numpy as np
import random
import math
from math import sqrt, cos, sin, tan, exp
import deap.benchmarks as bench # various test functions
from copy import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

verbose = False
verbose2 = verbose or True
firstTargetRatio = 0.9
scal = 0.15 #greediness
pseudo = 0.0001

# These are the functions Simon defined to test it on:
def simon_f1(x, y):
    return 0.2 * (x**2 + y**2) / 2 + 10 * sin(x+y)**2 + 10 * sin(100 * (x - y))**2

def simon_f2(xs):
    x, y = xs
    return simon_f1(x - 100, y - 100)

#Find the n smallest values in xs
def mins(xs_, n):
    xs = list(xs_) # copy xs
    nc = 0 # counter for the number of small values
    while nc < n: # now we essentially do selection sort
        for i in range(len(xs) - 1):
            for j in range(i + 1, len(xs)):
                if xs[j] < xs[i]:
                    t = xs[i]
                    xs[i] = xs[j]
                    xs[j] = t
                    nc += 1
                    break

    return xs[:n]

#Find the n smallest values in xs, comparing by a given function
def minsf(xs_, n, f):
    xs = list(xs_) # copy xs
    nc = 0 # counter for the number of small values
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            if f(xs[j]) < f(xs[i]):
                t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
                nc += 1
        if nc == n: break

    return xs[:n]

def fst(xs):
    return xs[0]

def snd(xs):
    return xs[1]

#Get 2 values in the list that are nearest to the target.
# Returns a list of tuples [(x, distance)]
def nears(xs, target):
    if len(xs) == 0: #Can't deal with emptiness.
        raise "Empty list"
    elif len(xs) == 1: #Only one option.
        return xs[0]

    if verbose: print("Finding nearest in list of " + str(len(xs)) + " elements.")

    # ns will hold our 2 values. We simply pick the two first ones from the input list.
    ns = map((lambda x: (x, norm(target - x))), xs[:2])

    for x in xs:
        delta = norm(target - x)
        if delta < ns[0][1]: # ns[0][0] will be *the* closest point.
            ns[0] = (x, delta)
        elif delta < ns[1][1]: # ns[1][0] will be the second closest point.
            ns[1] = (x, delta)

    return ns

def norm(v):
    return sqrt(sum(v**2))

def newTarget(vals, best):
    # get the minimum y value recorded in our list of minima, and push down from there.
    return best + scal * (best - min(map((lambda v: v[0]), vals)))

# move the target lower, by adding to it the scaled difference between the two best
# minima so far found.
def refreshTarget(vals):
    # get the two best minima so far found
    bests = minsf(vals,2,fst) #compare on y value in vals. Recall: vals :: [(y,x)]
    # bests :: [(y,x)], too !
    if verbose2: print("Two best minima so far: ", bests[0], " and ", bests[1])
    diff = bests[0][0] - bests[1][0]
    if verbose2: print("Y-Difference between the minima: ", diff)
    if diff > 0:
        print("Warning: difference between two best minima is nonnegative!")
    return bests[0][0] + scal * (-abs(diff))

# perform the optimization
def down(fun, niter, tol, x0, x1, dim):
    if verbose: print("Starting positions: " + str(x0) + ", " + str(x1) + ".")

    #fcs = 0 # total number of function calls required.

    # mathematica FindMinimum returns {f(x), x}, whereas fmin returns {x, f(x)}.
    # So simon's sol1[1] is fun(x0) in this case.
    xv1 = fmin(fun, x0, disp=verbose)
    fv1 = fun(xv1)
    if verbose:
        print("first minimum value: " + str(fun(x0)))
        print("Second minimum value: " + str(fv1))

    vals = [(fun(x0), x0), (fv1, xv1)]
    target = firstTargetRatio * fv1
    if verbose2: print("Initialized target to " + str(target))

    nx1 = x1
    lpos = [nx1]

    if verbose: print("Entering loop.")

    for i in xrange(niter):
        print("Guess point for this iteration: ", nx1)
        pmin = fmin(fun, nx1, disp=verbose) #pmin is the x of the local minimum found in this iteration
        fv = fun(pmin)
        if verbose: print("Found new local minimum f", pmin, " = ", fv, sep='')
        delta = fv - target # how far are we from the target y value
        if verbose: print("Got delta: " + str(delta))

        if delta < 0: # we've fallen below the target, so we must update it.
            target = newTarget(vals, fv)
            if verbose: print("Target updated to " + str(target))
            delta = fv - target #recalculate delta for the new target
            if verbose: print("New delta: " + str(delta))

        if verbose: print("Target y-value ", target)

        near = nears(map((lambda x: x[1]), vals), pmin) #map to extract x values (vals :: [(y,x)])
        # near :: [(x, distance)]
        if verbose: print("Got near values " + str(near))

        if near[0][1] < tol:
            if verbose: print("Closest old minimum is too close. Choosing second-closest.")
            nnear = near[1][0]
        else:
            nnear = near[0][0]

        fnear = fun(nnear) # function-value at the minimum nearest to pmin

        if verbose: print("Nearest old minimum f", nnear, " = ", fnear, sep='')

        deltan = fnear - target # how far away from the target are we
        if verbose: print("Got deltan: " + str(deltan))

        if (delta - deltan)**2 < pseudo**2:
            step = -delta / pseudo * (pmin - nnear)
        else:
            step = -delta / (delta - deltan) * (pmin - nnear)

        if verbose: print("Step endpoints: ", pmin, " and ", nnear)

        # if verbose: print("Got step " + str(target)) # we don't need this *again*...

        nx1 += step
        lpos.append(copy(nx1))
        if verbose2: print("Took step ", step)

        vals.append((fv, pmin))

        if norm(step) < tol:
            if verbose2: print("Found fixed point: f", pmin, " = ", nnear, sep='')
            return pmin, fv, lpos
        if i % 10 == 0: # A CONSTANT ! Consider changing this.
            oldtarget = target
            target = refreshTarget(vals)
            if target > oldtarget:
                print("Target increased! Undoing refresh.")
                target = oldtarget
            if verbose2: print("Refreshed target to " + str(target))
            # if ((oldtarget - target) / target)**2 > 0.01: # i.e. 0.1**2 # why?

    return sorted(vals)[0], fv, lpos

def randomGuess(dim, scale = 1):
    return np.array([scale * random.uniform(0,2) for q in xrange(dim)])

# deap's benchmarking functions return the values as tuples with nothing in element 1,
# so we need to unpack element 0 which is the actual function-value.
def unwrapBench(f):
    return lambda x: f(x)[0]

def test(f):
    xmin, fv, lpos = down(f, 2500, 0.1, randomGuess(2, 10), randomGuess(2, 10), 2)
    print("Found global minimum f", xmin, " = ", fv, sep='')
    print("Steps taken: ")
    map((lambda p: print(" ", p)), lpos)

def plotf(f, start=-10, end=10, smoothness=0.4):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xs = ys = np.arange(start, end, smoothness)
    X, Y = np.meshgrid(xs, ys)
    zs = np.array([f((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
