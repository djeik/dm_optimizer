#!/usr/bin/env python

from __future__ import print_function

import sys
import heapq
import numpy as np
import scipy as sp
import scipy.optimize as opt

from itertools import repeat
from dm_utils import random_vector

# Different verbosity settings.
QUIET = (None, -1)
ERROR = ("ERROR", 0)
WARN  = ("WARNING", 1)
INFO  = ("INFO", 2)
DEBUG = ("DEBUG", 3)
NOISE = ("NOISE", 4)

# Default verbosity is to show only error messages.
verbosity = ERROR

def log(priority, *args, **kwargs):
    """ Forward arguments and keyword arguments to the print function only if
        the given message priority is at most the global verbosity level.
        Priorities are tuples of a string naming that priority and an integer
        representing the level of urgency of the message, where zero is the
        most urgent (error messages) and greater values are of decreasing
        importance.
        """
    if priority[1] <= verbosity[1]:
        print(priority[0], *args, **kwargs)

def elog(priority, *args, **kwargs):
    """ Convenience function to log to standard error. By default `log` logs to
        standout output.
        """
    log(priority, *args, file=sys.stderr, **kwargs)

def bh(fun, niter, dim=2, distance=1):
    x1 = random_vector(dim, distance)
    r = opt.basinhopping(fun, x1)
    r.x = list(r.x)
    return dict(r)

def dm(fun, niter, tol=1e-8, dim=2, firsttargetratio=0.9, scal=0.05,
        pseudo=1e-4, refresh_rate=10, distance=1):
    def newtarget(minima, best):
        """ Create a new target value, either initially to start the
            optimization, or later if the target is beaten.
            """
        return best[0] + scal * (best[0] - min(m[0] for m in minima))

    def refreshtarget(minima):
        """ Prevent the target from becoming stale.
            Called after adding the current local minimum to the list of
            minima.
            """
        best, second_best = heapq.nsmallest(2, minima, key=lambda m: m[0])
        return best[0] + scal * (best[0] - second_best[0])

    log = elog # log to standard error.

    x0, x1 = random_vector(dim, distance), random_vector(dim, distance)

    log(INFO, "starting positions: \n\tx0 = ", x0, "\n\tx1 =", x1, sep='')

    opt_result = opt.minimize(fun, x0)
    local_min = opt_result.fun, opt_result.x
    minima = [ ( fun(x0), x0 ), local_min ]

    nfev = 1 + opt_result.nfev

    target = firsttargetratio * local_min[0]

    log(INFO, "target initialized to: \n\ty =", target)

    iterate = x1
    iterate_positions = [ x1 ]

    log(DEBUG, "beginning optimization loop")

    i = 0
    for i in xrange(niter):
        log(DEBUG, "iteration", i)

        opt_result = opt.minimize(fun, iterate)
        nfev += opt_result.nfev
        local_min = (opt_result.fun, opt_result.x)

        log(INFO, "found local minimum: \n\tx = ", local_min[1],
                "\n\ty =", local_min[0], sep='')

        delta = local_min[0] - target

        if delta < 0:
            # then we have beaten the target, and need to create a new one.
            target = newtarget(minima, local_min)
            log(INFO, "target updated to: \n\ty =", target)
            delta = local_min[0] - target

        # Get the two past minima that are closest to the current local
        # minimum. We will choose just one based on tolerances.
        nearest = heapq.nsmallest(2, minima,
                key=lambda m: np.linalg.norm(m[1] - local_min[1]))

        if np.linalg.norm(nearest[0][1] - local_min[1])**2 < tol**2:
            # If the nearest minimum is *too* close, then take the second
            # nearest one instead.
            nearest = nearest[1]
            log(DEBUG, "selecting second nearest past minimum instead of "
                    "true nearest")
        else:
            # Otherwise we take the true nearest minimum as the one on
            # which to perform the next computations.
            nearest = nearest[0]

        log(DEBUG, "nearest past minimum: \n\tx = ", nearest[1],
                "\n\ty =", nearest[0], sep='')

        # the distance to the target from the nearest past local minimum
        delta_near = nearest[0] - target

        if (delta - delta_near)**2 < pseudo**2:
            step = (-delta / pseudo) * (local_min[1] - nearest[1])
        else:
            step = -delta / (delta - delta_near) * (local_min[1] - nearest[1])

        log(INFO, "calculated step: \n\tdx =", step)

        iterate = iterate + step

        minima.append(local_min)
        iterate_positions.append(iterate)

        if np.linalg.norm(step) < tol:
            log(INFO, "found fixed point: \n\tx = ", local_min[1],
                    "\n\tx_near = ", nearest[1], sep='')
            return {
                "x": list(local_min[1]),
                "fun": local_min[0],
                "status": 0,
                "success": True,
                "niter": i,
                "nfev": nfev,
                "message": [ "Fixed point found" ],
                "iterate": list(list(ip) for ip in iterate_positions),
                "minima": list(map(lambda (a, b): (a, list(b)), minima)),
            }

        if i % refresh_rate == 0:
            oldtarget = target
            target = refreshtarget(minima)

            if target != 0 and ((oldtarget - target) / target)**2 > 0.1**2: # TODO const
                log(INFO, "refreshed target: \n\tt =", target)

    (y, x) = min(minima, key=lambda m: m[0])

    return {
        "x": list(x),
        "fun": y,
        "messages": ["the requested number of iterations completed successfully"],
        "success": True,
        "iterate": list(list(ip) for ip in iterate_positions),
        "minima": list(map(lambda (a, b): (a, list(b)), minima)),
        "nfev": nfev,
        "niter": i,
        "status": 1
    }
