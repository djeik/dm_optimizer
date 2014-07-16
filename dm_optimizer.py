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
            verbosity           -- controls debug output; higher numbers = more messages.
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

        if not target is None:
            self.logmsg(7, "Setting fixed target.")
            self.fixed_target = True
            self.target       = target
        else:
            self.fixed_target = False


    def logmsg(self, priority, *args, **kwargs):
        """ Forward all arguments to print if the given priority is less than or equal to the verbosity of the optimizer. The file keyword argument
            of print is set to sys.stderr.

            Arguments:
                priority -- if this value is less than `verbosity`, all other arguments are forwarded to print.

            Returns:
                Nothing.
            """
        if priority == self.verbosity:
            print(*args, file=self.logfile, **kwargs)

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
                mins = [self.vals[0]]
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

    def very_greedy_refresh_target(self):
        """ Perform a target refresh with the best and worst local minima only, rather than the two best. """
        if self.fixed_target:
            pass
            self.logmsg(2, "Target is fixed; refusing to refresh target.")
        else:
            if len(self.vals) < 2:
                raise ValueError("Insufficient number of discovered local minima to perform a target update.")
            return self.vals[0].y + self.greediness * (self.vals[0].y - self.vals[-1].y)

    def step_to_nearest_minimum(self, (ynear, xnear), deltay_curr):
        """ Create a 'step' delta x in the direction of the nearest past local minimum. """

        #if self.iteration % self.refresh_rate == 0:
        #    self.logmsg(7, "Stepping towards best minimum.")
        #    stepdir = self.vals[0].x - self.pmin
        #else:
        stepdir = xnear - self.pmin # the direction of the step

        deltay_prev = ynear - self.target # how far away is the best previously-found local minimum from the target
        # if things are going well, then deltay_curr should be less than deltay_prev
        self.logmsg(2, "Best minimum delta:", deltay_prev)

        if (deltay_curr - deltay_prev)**2 < self.pseudo**2: # calculate the step scale.
            self.logmsg(1, "Very close deltas.")
            stepscale = deltay_curr / self.pseudo # Use pseudo to avoid jumping away to infinity, if necessary.
        else:
            stepscale = deltay_curr / (deltay_curr - deltay_prev)

        #if abs(stepscale) < self.tolerance:
        #    stepscale = stepscale / abs(stepscale) * 2.0

        step = stepscale * stepdir

        if stepscale == 0:
            raise Exception("Zero length step towards nearest minimum.")

        self.logmsg(2, "Current delta y:", deltay_curr)
        self.logmsg(2, "Stepscale:", stepscale)
        self.logmsg(2, "Step:", step)
        self.logmsg(1, "Step size:", norm(stepdir))

        return step

    def step_to_best_minimum(self, deltay_curr):
        stepdir = self.vals[0].x - self.pmin
        deltay_prev = self.vals[0].y - self.target
        #if deltay_prev < 0:
        #    raise ValueError("Negative y-distance to target.")

        if (deltay_curr - deltay_prev)**2 < self.pseudo**2:
            stepscale = deltay_curr / self.pseudo
        else:
            stepscale = deltay_curr / (deltay_curr - deltay_prev)

        step = stepscale * stepdir

        return step

    def average_minima_spread(self):
        """ For each minimum, calculate its average distance to all the other minima, and average these averages.
            This gives a measure of how much of the search space was explored.
            """
        totals = []
        for (i, mini) in enumerate(islice(self.vals, 0, len(self.vals)-1)):
            norms = map(lambda minj: norm(mini.x - minj.x), islice(self.vals, i+1))
            totals.append(sum(norms) / len(norms))
        return sum(totals) / len(totals)

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

        if not self.callback is None:
            self.vs = []

        # and that our counters are zero
        self.nfev       = 0
        self.njev       = 0
        self.nhev       = 0
        self.iteration  = 0

        # Get the first minimum
        x0min = self.local_min(x0)
        f0 = self.evalf(x0min)

        self.vals.add(value_box( (f0, x0min) ))
        self.valsi.append(value_box( (f0, x0min) ))

        if not self.fixed_target:
            self.target = self.first_target_ratio * f0

        nx1 = x1
        self.fv = f0
        self.pmin = x0min
        self.lpos = [(self.evalf(nx1), nx1)]

        self.logmsg(2, "Entering loop.")

        try:
            for self.iteration in xrange(self.max_iterations):
                self.logmsg(2, "Guess point for this iteration:", nx1)
                self.pmin = self.local_min(nx1)
                self.fv = self.evalf(self.pmin) # the function value at the local minimum for this iteration
                self.logmsg(6, "Minimum for this iteration f", self.pmin, " = ", self.fv, sep='')
                self.logmsg(7, "Target:", self.target)
                deltay_curr = self.fv - self.target # how far away is our current minimum from the target value

                self.logmsg(2, "Got delta:", deltay_curr)

                #if deltay_curr <= self.tolerance**2: # we've fallen below the target, so we create a new target
                if deltay_curr < 0: # we've fallen below the target, so we create a new target
                    self.logmsg(2, "Beat target!")
                    self.new_target(self.fv) # to get the next target, all we need is the current function value and the list of minima so far
                    deltay_curr = self.fv - self.target # recalculate delta for the new target
                    self.logmsg(2, "New delta:", deltay_curr)
                else:
                    pass
                    self.logmsg(2, "Target y-value", self.target)

                if deltay_curr < self.tolerance:
                    #raise Exception("Current distance to target is too small; target update failed.")
                    deltay_curr = self.tolerance # this is probably not wise.

                # let's take the *best* minimum, i.e. the minimum closest to our current minimum
                try:
                    ynear, xnear = self.best_minimum_x().unbox()
                except BestMinimumException:
                    res = sopt.OptimizeResult()
                    res.nit     = self.iteration
                    res.success = True
                    res.message = ["All local minima have converged to a point. Optimization cannot proceed."]
                    res.status  = 3
                    res.x       = self.pmin
                    res.fun     = self.fv
                    res.njev    = self.njev
                    res.nfev    = self.nfev
                    res.lpos    = self.lpos
                    res.opt     = self
                    return res


                self.logmsg(1, "Nearest old minimum f", xnear, " = ", ynear, sep='')

                if self.iteration % self.refresh_rate == 0:
                    self.step = self.step_to_best_minimum(deltay_curr)
                else:
                    self.step = self.step_to_nearest_minimum( (ynear, xnear), deltay_curr)

                nx1 += self.step # actually take the step
                self.lpos.append((self.evalf(nx1), copy(nx1))) # add the new position to the list of past positions
                self.logmsg(2, "Took step ", self.step)

                self.vals.add(value_box( (self.fv, copy(self.pmin)) ))
                self.valsi.append(value_box( (self.fv, copy(self.pmin)) ))
                map(lambda x: self.logmsg(2, x.unbox()), self.vals[-3:])

                if norm(self.step) < self.tolerance:
                    self.logmsg(1, "Found fixed point: f", self.pmin, " = ", ynear, sep='')
                    self.logmsg(1, "Step was:", self.step)
                    newtarget1 = self.refresh_target()
                    newtarget2 = self.very_greedy_refresh_target()
                    self.target = newtarget2
                    self.logmsg(7, "Difference in targets (fixed-point): ", newtarget1 - newtarget2)

                # every `refresh_rate` iterations, we refresh the target value, to avoid stale convergeance
                if (not self.fixed_target) and self.iteration % self.refresh_rate == 0:
                    m = self.chopfactor
                    del self.vals[m:-m]
                    oldtarget = self.target
                    newtarget1 = self.refresh_target() # mutates `target`
                    newtarget2 = self.very_greedy_refresh_target()
                    self.target = newtarget2
                    self.logmsg(7, "Difference in targets (afterdel): ", newtarget1 - newtarget2)

                if not self.callback is None:
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
        return optimizer.minimize(x1, x2)
    except Exception as e:
        res = sopt.OptimizeResult()
        res.message = [str(e)]
        res.status  = 2
        res.success = False
        res.opt = optimizer # we'll put the broken optimizer in here in case we need it to recover some failure information
        return res

