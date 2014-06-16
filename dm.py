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

verbosity = 1
firstTargetRatio = 0.9
scal = 0.15 #greediness
pseudo = 0.0001

def logmsg(priority, *args, **kwargs):
    if priority <= verbosity:
        print(*args, **kwargs)
    else:
        pass

# These are the functions Simon defined to test it on:
def simon_f1(xy):
    x, y = xy
    return 0.2 * (x**2 + y**2) / 2 + 10 * sin(x+y)**2 + 10 * sin(100 * (x - y))**2

def simon_f2(xs):
    xy = xs - [100, 100]
    return simon_f1(xy)

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

    logmsg(2, "Finding nearest in list of " + str(len(xs)) + " elements.")

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
    bests = mins(vals,2,fst) #compare on y value in vals. Recall: vals :: [(y,x)]
    # bests :: [(y,x)], too !
    logmsg(2, "Two best minima so far: ", bests[0], " and ", bests[1])
    diff = bests[0][0] - bests[1][0]
    logmsg(2, "Y-Difference between the minima: ", diff)
    return bests[0][0] + scal * (-abs(diff))

def bestMinimum(prev_minima, target, fv, tol):
    (y1, x1), (y2, x2) = mins(prev_minima, 2, comp=lambda (y, x): y - target)
    if abs(y1-fv) < tol: # if the *best* minimum is the *current* minimum (fixed-point like behaviour)...
        return (y2, x2) # return the *second*-best minimum.
    else:
        return (y1, x1)

# perform the optimization
def down(fun, niter, tol, x0, x1):
    logmsg(2, "Starting positions: " + str(x0) + ", " + str(x1) + ".")

    #fcs = 0 # total number of function calls required.

    # mathematica FindMinimum returns {f(x), x}, whereas fmin returns {x, f(x)}.
    # So simon's sol1[1] is fun(x0) in this case.
    x0min = fmin(fun, x0, disp=(verbosity==2))
    f0 = fun(x0min)
    x1min = fmin(fun, x1, disp=(verbosity==2))
    f1 = fun(x1min)

    logmsg(1, "first minimum value: ", f0)
    logmsg(1, "Second minimum value: ", f1)

    vals = [(f1, x1min), (f0, x0min)]
    target = firstTargetRatio * f1
    logmsg(1, "Initialized target to " + str(target))

    nx1 = x1
    lpos = [(fun(nx1), nx1)]

    logmsg(2, "Entering loop.")

    try:
        for i in xrange(niter):
            logmsg(2, "Guess point for this iteration: ", nx1)
            pmin = fmin(fun, nx1, disp=(verbosity==2)) #pmin is the x of the local minimum found in this iteration
            fv = fun(pmin) #fv is the function-value at that local minimum
            logmsg(2, "Found new local minimum f", pmin, " = ", fv, sep='')

            delta = fv - target # how far are we from the target y value

            logmsg(2, "Got delta: " + str(delta))

            if delta < 0: # we've fallen below the target, so we must update it.
                target = newTarget(vals, fv)
                logmsg(1, "Target updated to " + str(target))
                delta = fv - target #recalculate delta for the new target
                logmsg(2, "New delta: " + str(delta))

            logmsg(2, "Target y-value ", target)

            ## wait this is no good...
            #fnear, nnear = nearestMinimum(vals, pmin, tol, fun)

            # let's take the *best* minimum instead.
            ynear, xnear = bestMinimum(vals, target, fv, tol)

            logmsg(2, "Nearest old minimum f", xnear, " = ", ynear, sep='')

            if fv < ynear: #if the current local minimum is *better* that any previous one, we move *away* from the old one
                logmsg(1, "Doubling back!")
                stepdir = xnear - pmin
            else: # otherwise, we move *towards* the old one
                stepdir = pmin - xnear

            deltay_prev = ynear - target # how far away is the best previously-found local minimum
            step = -delta / (pseudo if (delta - deltay_prev)**2 < pseudo**2 else (delta - deltay_prev)) * stepdir

            logmsg(2, "Step endpoints: ", pmin, " and ", xnear)

            nx1 = nx1 + step
            lpos.append((fun(nx1), copy(nx1)))
            logmsg(2, "Took step ", step)

            vals.append((fv, pmin))

            if norm(step) < tol:
                logmsg(1, "Found fixed point: f", pmin, " = ", ynear, sep='')
                return pmin, fv, lpos
            elif norm(step) > 100:
                logmsg(1, "Taking very large step!")
                logmsg(1, "delta = ", delta, "; deltay_prev = ", deltay_prev, "; pmin = ", pmin, "; xnear = ", xnear, sep='')

            if i % 10 == 0: # A CONSTANT ! Consider changing this.
                oldtarget = target
                target = refreshTarget(vals)
                # target increase can occur during the first iteration or so.
                logmsg(1, "Refreshed target to " + str(target))
                # if ((oldtarget - target) / target)**2 > 0.01: # i.e. 0.1**2 # why?
    except KeyboardInterrupt:
        # Note: when running interactively, the second keyboard interrupt caught will just kill the interpreter.
        # note2: ipython doesn't die the second time.
        print("Halting optimization.")

    return sorted(vals)[0], fv, lpos

def randomGuess(dim, scale = 1):
    return np.array([scale * random.uniform(0,2) for q in xrange(dim)])

# deap's benchmarking functions return the values as tuples with nothing in element 1,
# so we need to unpack element 0 which is the actual function-value.
def unwrapBench(f):
    return lambda x: f(x)[0]

def test(f):
    xmin, fv, lpos = down(f, 2500, 0.0001, randomGuess(2, 10), randomGuess(2, 10))
    print("Found global minimum f", xmin, " = ", fv, sep='')
    print("Steps taken: ")
    map((lambda p: print(" ", p)), lpos)
    if len(xmin) == 2: #only plot if in 3D !
        plotf(f, lpos, -10, 10)

def testAckley():
    test(unwrapBench(bench.ackley))

def testSimon1():
    test(simon_f1)

# A visual debug tool, basically.
def plotf(f, xyzs_, start=-10, end=10, smoothness=0.3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xs = ys = np.arange(start, end, smoothness)
    X, Y = np.meshgrid(xs, ys)
    zs = np.array([f((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, # draw the surface
            linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    z, xys = zip(*xyzs_) # prepare to draw the line
    x, y = zip(*xys)
    N = len(x)
    for i in xrange(N-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.jet(255*i/N))
    plt.show()
