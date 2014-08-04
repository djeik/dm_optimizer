from __future__ import print_function

import scipy.optimize as sopt
import scipy
import numpy as np
from numpy.linalg import norm
from copy import copy

import sortedcontainers as sl
from functools import total_ordering
from itertools import imap, islice

import sys

class BestMinimumException(Exception):
    def BestMinimumException(*args, **kwargs):
        super(*args, **kwargs)

@total_ordering
class value_box:
    """ A silly class that represents points (y, x) compared on the basis of their y-components only. """
    def __init__(self, (y, x)):
        self.y = y
        self.x = x

    def unbox(self):
        return (self.y, self.x)

    def __lt__(self, other):
        return self.y < other.y

    def __eq__(self, other):
        return self.y == other.y

class dm_optimizer:
    """ A global optimizer using the difference map-based algorithm with or without constraints.
        Attributes:
            max_iterations      -- the maximum number of iterations to perform before aborting minimization.
            tolerance           -- the threshold above which we consider two points to be distinct.
            verbosity           -- controls debug output; higher numbers = more messages. Special string 'any' will cause everything to be logged.
            first_target_ratio  -- determines how low the first target should be.
            greediness          -- how quickly should the target value be lowered.
            fun                 -- the objective function.
            pseudo              -- special value to use to determine the step size when it would otherwise be very large.
            refresh_rate        -- how frequently to update the target value, in iterations per update.
            nfev                -- total number of function evaluations.
            vals                -- the local minima collected along the way, in ascending order.
            valsi               -- the local minima collected along the way, in order of discovery.
            lpos                -- the path the optimizer has followed along the objective function.
            callback            -- a function called as a method at the end of every iteration. Passed to it is a reference to the optimizer.
            minimizer_kwargs    -- keyword arguments passed to the local minimizer.
            logfile             -- (a file handle) where debug messages should be logged. (Default: sys.stderr)

        Notes:
            vals is stored as a sorted list, since we frequently just need to get the n smallest values.
            It is a list of tuples (y, x), which is convenient since Python tuples are totally ordered according
            to their first element.
            """

    def __init__(self, fun, max_iterations=2500, target=None, constraints=[], nonsatisfaction_penalty=0, tolerance=0.000001, verbosity=0,
            pseudo=0.001, greediness=0.05, first_target_ratio=0.9, refresh_rate=10, chopfactor=1, logfile=sys.stderr,
            callback=None, minimizer_kwargs={}):
        self._fun                       = fun # store the objective function in a 'hidden' attribute to avoid accidentally
                                              # calling the unwrapped version
        self.max_iterations             = max_iterations
        self.constraints                = constraints
        self.nonsatisfaction_penalty    = nonsatisfaction_penalty
        self.tolerance                  = tolerance
        self.verbosity                  = verbosity
        self.pseudo                     = pseudo
        self.greediness                 = greediness
        self.first_target_ratio         = first_target_ratio
        self.refresh_rate               = refresh_rate
        self.chopfactor                 = chopfactor
        self.logfile                    = logfile
        self.minimizer_kwargs           = minimizer_kwargs
        self.callback                   = callback

        if target is not None:
            self.logmsg(7, "Setting fixed target.")
            self.fixed_target = True
            self.target       = target
        else:
            self.fixed_target = False


    def logmsg(self, priority, *args, **kwargs):
        """ Forward all arguments to print if the given priority is less than or equal to the verbosity of the optimizer. The file keyword argument
            of print is set to the logfile.

            Arguments:
                priority -- if this value is less than `verbosity`, all other arguments are forwarded to print.

            Returns:
                Nothing.
            """
        if (isinstance(self.verbosity, str) and self.verbosity == "any") or priority == self.verbosity:
            print("%" + str(priority) + "%", *args, file=self.logfile, **kwargs)

    def fun(self, x):
        """ Evaluate the objective function at the given point and increment the running count of function calls. """
        self.nfev += 1
        return self._fun(x)

    def evalf(self, x):
        """ Evaluate the objective function at the given point and apply the nonsatisfaction penalty times the number of
            constraints not satisfied. """
        return self.fun(x) + sum([self.nonsatisfaction_penalty for constraint in self.constraints if not constraint(x)])

    def local_min(self, x):
        """ Find the local minimum nearest to the given point. """
        #return fmin(self.fun, x, xtol=self.tolerance, maxiter=2000, maxfun=2000, disp=self.verbosity==2)
        res = sopt.minimize(self.fun, x, **self.minimizer_kwargs)
        # it is unnecessary to manually increase self.nfev since self.fun already does this for us.
        # still, we must manually increase self.njev or self.nhev since minimize might use the Jacobian or Hessian
        if 'njev' in res.keys():
            self.njev += res.njev
        if 'nhev' in res.keys():
            self.nhev += res.nhev
        return res.x

    def best_minimum_x(self):
        """ Find the minimum in the list of past minima that is closest to the given value.

            Arguments:
                pmin -- the point from which the distances are to be calculated.

            Returns:
                _(y, x)_ -- the minimum closest to the given point.

            Notes:
                To avoid accidental rapid convergeance to a fixed point, this function will not return the true closest minimum
                if it is considered to be the same as the given point; it will instead return the second-closest minimum. Two points
                are considered the same if their distance is less than `tolerance`.
            """
        minima = sorted([x for x in self.vals
                                 if norm(x.x - self.pmin) >= self.tolerance
                                 #and (self.evalf(self.pmin) - x.y)**2 >= 0.1
                        ], key=lambda v: norm(self.pmin - v.x))

        if len(minima) == 0:
            raise BestMinimumException("There are no local minima far away enough from the current minimum to be considered distinct.")

        return minima[0]

    def new_target(self, best):
        """ Calculate a new target value.

            Arguments:
                best -- the current function-value.

            Returns:
                Nothing.

            Notes:
                Sets the objects internal `target` attribute to the new target value. This method should be called when the actual function value
                falls below the current target value.
            """
        if self.fixed_target:
            pass
            logmsg(7, "Target is fixed; refusing to create a new target.")
        else:
            # get the minimum y value recorded in our list of minima, and push down from there.
            oldtarget = self.target
            mins = []
            for v in self.vals:
                if v.y - best > self.tolerance:
                    mins.append(v.y)
                else:
                    self.logmsg(1, "Skipping over bad minimum in newtarget calculation.")
            if not mins:
                raise BestMinimumException()
                #mins = [self.vals[0].y]
            self.target = best + self.greediness * (best - min(mins))
            if self.target != oldtarget:
                pass
                self.logmsg(7, "Target updated to ", self.target, " (", "higher" if self.target > oldtarget else "lower", ")", sep='')

    def refresh_target(self): # no effects on attributes.
        """ Refresh the target value.

            Arguments:
                None.

            Returns:
                Nothing.

            Notes:
                Uses the internally-maintained list of past minima to set `target` to the new target value.
            """
        if self.fixed_target:
            pass
            self.logmsg(7, "Target is fixed. Refusing to refresh target.")
        else:
            if len(self.vals) < 2:
                raise ValueError("Insufficient number of discovered local minima to perform a target update.")

            self.logmsg(1, "Two best minima so far: ", self.vals[0].unbox(), " and ", self.vals[1].unbox())
            diff = self.vals[0].y - self.vals[1].y
            if diff > 0: # this should never evaluate to True, since self.vals is sorted in ascending order.
                raise Exception("The difference between the two best minima should be negative!")
            self.logmsg(2, "Y-Difference between the minima: ", diff)
            self.logmsg(1, "Refreshed target to", self.target)

            return self.vals[0].y + self.greediness * diff

    def calculate_step_scale(self, destination, deltay_curr):
        return 0.75

    def step_toward(self, destination, deltay_curr):
        """ Calculate a step toward a given destination using the standard stepscale calculation method.
            If the destination's function-value is less good than the current one, the direction is reversed.
            """
        stepdir = (1 if destination.y <= self.fv else -1) * (destination.x - self.pmin)
        return self.calculate_step_scale(destination, deltay_curr) * stepdir

    def step_to_best_minimum(self, deltay_curr):
        return self.step_toward(vals[0], deltay_curr)

    def fv_after_step(self, step):
        """ Evaluate the score of the objective function after hypothetically taking the given step. """
        return self.evalf(self.nx1 + step)

    def all_possible_steps(self, deltay_curr):
        return map(lambda x: self.step_toward(x, deltay_curr), self.vals)

    def best_of_steps(self, steps):
        return min(steps, key=self.fv_after_step)

    def best_possible_step(self, deltay_curr):
        return self.best_of_steps(self.all_possible_steps(deltay_curr))

    def average_minima_spread(self):
        """ For each minimum, calculate its average distance to all the other minima, and average these averages.
            This gives a measure of how much of the search space was explored.
            """
        totals = []
        for (i, mini) in enumerate(islice(self.vals, 0, len(self.vals)-1)):
            norms = map(lambda minj: norm(mini.x - minj.x), islice(self.vals, i+1))
            totals.append(sum(norms) / len(norms))
        return sum(totals) / len(totals)

    def take_step(self, step):
        self.nx1 += step
        self.step = step

    def minimize(self, x0, x1):
        """ Perform the minimization given two initial points.

            Arguments:
                x0, x1 -- the initial points.

            Returns:
                _lpos_ -- the points the optimizer travelled to, the last of which is the global minimum or
                          the best minimum if the exhaustion.

            Notes:
                If the number of iterations exceeds the value set in the constructor, then the best local minimum found so far is returned.
            """
        # Ensure that our lists are empty.
        self.vals                       = sl.SortedList()
        self.valsi                      = []
        self.lpos                       = []

        if self.callback is not None:
            self.vs = []

        # and that our counters are zero
        self.nfev       = 0
        self.njev       = 0
        self.nhev       = 0
        self.iteration  = 0

        # and the some other things are given zero initial values
        self.step       = []

        # Get the first minimum
        x0min = self.local_min(x0)
        f0 = self.evalf(x0min)

        self.vals.add(value_box( (f0, x0min) ))
        self.valsi.append(value_box( (f0, x0min) ))

        if not self.fixed_target:
            self.target = self.first_target_ratio * f0

        self.nx1 = x1
        self.fv = f0
        self.pmin = x0min
        self.lpos = [(self.evalf(self.nx1), self.nx1)]

        self.logmsg(2, "Entering loop.")

        try:
            for self.iteration in xrange(1, self.max_iterations + 1):
                self.logmsg(2, "Guess point for this iteration:", self.nx1)
                self.pmin = self.local_min(self.nx1)
                self.fv = self.evalf(self.pmin) # the function value at the local minimum for this iteration
                self.logmsg(6, "Minimum for this iteration f", self.pmin, " = ", self.fv, sep='')
                self.logmsg(7, "Target:", self.target)
                deltay_curr = self.fv - self.target # how far away is our current minimum from the target value

                self.logmsg(2, "Got delta:", deltay_curr)

                #if deltay_curr <= self.tolerance**2: # we've fallen below the target, so we create a new target
                if deltay_curr < 0: # we've fallen below the target, so we create a new target
                    self.logmsg(2, "Beat target!")
                    try:
                        self.new_target(self.fv) # to get the next target, all we need is the current function value and the list of minima so far
                    except BestMinimumException:
                        # this is a failure sink
                        res = sopt.OptimizeResult()
                        res.nit     = self.iteration
                        res.success = True
                        res.message = ["All local minima have converged to a point. Optimization cannot proceed.", "new_target failed."]
                        res.status  = 3
                        res.x       = self.pmin
                        res.fun     = self.fv
                        res.njev    = self.njev
                        res.nfev    = self.nfev
                        res.lpos    = self.lpos
                        res.opt     = self
                        return res

                    deltay_curr = self.fv - self.target # recalculate delta for the new target
                    self.logmsg(2, "New delta:", deltay_curr)
                else:
                    pass
                    self.logmsg(2, "Target y-value", self.target)

                if deltay_curr < self.tolerance:
                    #raise Exception("Current distance to target is too small; target update failed.")
                    deltay_curr = self.tolerance # this is probably not wise.

                self.take_step(self.best_possible_step(deltay_curr))

                self.lpos.append((self.evalf(self.nx1), copy(self.nx1))) # add the new position to the list of past positions
                self.logmsg(2, "Took step ", self.step)

                self.vals.add(value_box( (self.fv, copy(self.pmin)) ))
                self.valsi.append(value_box( (self.fv, copy(self.pmin)) ))
                map(lambda x: self.logmsg(2, x.unbox()), self.vals[-3:])

                if norm(self.step) < self.tolerance:
                    self.logmsg(1, "Found fixed point at", self.pmin)
                    self.logmsg(1, "Step was:", self.step)
                    newtarget1 = self.refresh_target()
                    self.target = newtarget1

                # every `refresh_rate` iterations, we refresh the target value, to avoid stale convergeance
                if (not self.fixed_target) and self.iteration % self.refresh_rate == 0:
                    oldtarget = self.target
                    newtarget1 = self.refresh_target() # mutates `target`
                    self.target = newtarget1

                if self.callback is not None:
                    self.callback(self)

        except Exception as e:
            self.logmsg(-1, "An error occurred while optimizing:", e)
            raise e

        # we reach this point if no exceptions have been raised and max_iterations has been exhausted
        self.lpos.append(min(self.vals, key=lambda v: v.y).unbox())
        self.fv, self.pmin = self.lpos[-1]

        res = sopt.OptimizeResult()
        res.nit     = self.iteration
        res.success = True
        res.message = ["Maximum number of iterations reached."]
        res.status  = 1
        res.x       = self.pmin
        res.fun     = self.fv
        res.njev    = self.njev
        res.nfev    = self.nfev
        res.lpos    = self.lpos
        res.opt     = self
        return res

def minimize(f, x1, x2, **kwargs):
    """ Construct a dm_optimizer object and run its minimize method with the given initial points.
        The result is packaged into a scipy OptimizeResult
        object.

        Arguments:
            f       -- the objective function.
            x1, x2  -- the initial points.
            kwargs  -- passed to the constructor of dm_optimizer

        Returns:
            A scipy.optimize.OptimizeResult object populated with typical data.
            Refer to the scipy documentation on OptimizeResult for more information.
        """
    optimizer = dm_optimizer(f, **kwargs)

    #return optimizer.minimize(x1, x2)
    try:
        res = optimizer.minimize(x1, x2)
    except Exception as e:
        res = sopt.OptimizeResult()
        res.message = ["Exception!", str(e)]
        res.status  = 2
        res.success = False
        res.nfev    = optimizer.nfev
        res.njev    = optimizer.njev
        res.opt = optimizer # we'll put the broken optimizer in here in case we need it to recover some failure information

    try:
        optimizer.logmsg(1, "Exit message(s):", *res.message)
    except Exception as e:
        print("Fatal error: could not write exit messages to logfile.")
    return res
